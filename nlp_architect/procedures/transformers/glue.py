# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import argparse
import io
import logging
import os
import numpy as np

from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nlp_architect.data.glue_tasks import get_glue_task, processors
from nlp_architect.models.transformers import TransformerSequenceClassifier
from nlp_architect.nn.torch import setup_backend, set_seed
from nlp_architect.procedures.procedure import Procedure
from nlp_architect.procedures.registry import (register_run_cmd,
                                               register_train_cmd)
from nlp_architect.procedures.transformers.base import (create_base_args,
                                                        inference_args,
                                                        train_args)
from nlp_architect.utils.io import prepare_output_path
from nlp_architect.utils.metrics import (acc_and_f1, pearson_and_spearman,
                                         simple_accuracy)
from zoo.pipeline.api.net.torch_net import TorchNet
from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy
from zoo.pipeline.api.keras.optimizers import AdamWeightDecay
from bigdl.util.common import Sample
from bigdl.optim.optimizer import DistriOptimizer, MaxEpoch, EveryEpoch

logger = logging.getLogger(__name__)


@register_train_cmd(name='transformer_glue',
                    description='Train (finetune) a BERT/XLNet/XLM model on a GLUE task')
class TransformerGlueTrain(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_glue_args(parser)
        create_base_args(parser, model_types=TransformerSequenceClassifier.MODEL_CLASS.keys())
        train_args(parser, models_family=TransformerSequenceClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_run_cmd(name='transformer_glue',
                  description='Run a BERT/XLNet/XLM model on a GLUE task')
class TransformerGlueRun(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_glue_args(parser)
        add_glue_inference_args(parser)
        inference_args(parser)
        create_base_args(parser, model_types=TransformerSequenceClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def add_glue_args(parser: argparse.ArgumentParser):
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: "
                        + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain dataset files to be parsed "
                        + "by the dataloaders.")


def add_glue_inference_args(parser: argparse.ArgumentParser):
    parser.add_argument("--evaluate", action='store_true',
                        help="Evaluate the model on the task's development set")


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    set_seed(args.seed, n_gpus)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    classifier = TransformerSequenceClassifier(model_type=args.model_type,
                                               model_name_or_path=args.model_name_or_path,
                                               labels=task.get_labels(),
                                               task_type=task.task_type,
                                               metric_fn=get_metric_fn(task.name),
                                               config_name=args.config_name,
                                               tokenizer_name=args.tokenizer_name,
                                               do_lower_case=args.do_lower_case,
                                               output_path=args.output_dir,
                                               device=device,
                                               n_gpus=n_gpus)

    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpus)

    train_ex = task.get_train_examples()
    dev_ex = task.get_dev_examples()
    train_dataset = classifier.convert_to_tensors(train_ex, args.max_seq_length)
    dev_dataset = classifier.convert_to_tensors(dev_ex, args.max_seq_length)
    total_steps, _ = classifier.get_train_steps_epochs(args.max_steps,
                                                       args.num_train_epochs,
                                                       args.per_gpu_train_batch_size,
                                                       len(train_dataset))

    sc = init_nncontext()
    torch_model = OuterBERT(classifier.model)
    net = TorchNet.from_pytorch(torch_model, [[2, 128], [2, 128], [2, 128]])
    train_samples = dataset_to_samples(train_dataset)
    dev_samples = dataset_to_samples(dev_dataset)
    train_rdd = sc.parallelize(train_samples, 4)
    dev_rdd = sc.parallelize(dev_samples, 4)
    optimizer = DistriOptimizer(net, train_rdd, SparseCategoricalCrossEntropy(),
                                MaxEpoch(args.num_train_epochs), 32,
                                AdamWeightDecay(lr=args.learning_rate,
                                                total=total_steps,
                                                weight_decay=args.weight_decay,
                                                epsilon=args.adam_epsilon))
    optimizer.set_validation(train_batch_size, dev_rdd, EveryEpoch())
    optimizer.optimize()

    # classifier.save_model(args.output_dir, args=args)


def dataset_to_samples(data_set):
    input_ids = data_set.tensors[0].numpy()
    input_mask = data_set.tensors[1].numpy()
    segment_ids = data_set.tensors[2].numpy()
    samples = []
    for i in range(0, len(input_ids)):
        # In BertForSequenceClassification, input order is input_ids, token_type_ids, attention_mask
        sample = Sample.from_ndarray([input_ids[i], segment_ids[i], input_mask[i]], 0)
        samples.append(sample)
    return samples


from torch import nn


class OuterBERT(nn.Module):
    def __init__(self, model):
        super(OuterBERT, self).__init__()
        self.bert = model

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        input_ids = input_ids.long()
        # if token_type_ids:
        token_type_ids = token_type_ids.long()
        # if attention_mask:
        attention_mask = attention_mask.long()
        return self.bert.forward(input_ids, token_type_ids, attention_mask, labels, position_ids, head_mask)


def do_inference(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, n_gpus)
    classifier = TransformerSequenceClassifier.load_model(model_path=args.model_path,
                                                          model_type=args.model_type,
                                                          metric_fn=get_metric_fn(task.name),
                                                          do_lower_case=args.do_lower_case)
    classifier.to(device, n_gpus)

    examples = task.get_dev_examples() if args.evaluate else \
        task.get_test_examples()
    data_set = classifier.convert_to_tensors(examples, include_labels=args.evaluate)
    samples = dataset_to_samples(data_set)

    sc = init_nncontext()
    torch_model = OuterBERT(classifier.model)
    net = TorchNet.from_pytorch(torch_model, [[2, 128], [2, 128], [2, 128]])
    preds = net.predict(sc.parallelize(samples, 4)).collect()
    preds = np.argmax(np.stack(preds), axis=1)
    classifier.evaluate_predictions(preds, data_set.tensors[3])
    with io.open(os.path.join(args.output_dir, "output.txt"), "w", encoding="utf-8") as fw:
        for p in preds:
            fw.write("{}\n".format(p))


# GLUE task metrics
def get_metric_fn(task_name):
    if task_name == "cola":
        return lambda p, l: {"mcc": matthews_corrcoef(p, l)}
    if task_name == "sst-2":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "mrpc":
        return acc_and_f1
    if task_name == "sts-b":
        return pearson_and_spearman
    if task_name == "qqp":
        return acc_and_f1
    if task_name == "mnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "mnli-mm":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "qnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "rte":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "wnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    raise KeyError(task_name)
