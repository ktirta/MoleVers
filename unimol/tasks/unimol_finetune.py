# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)

from unimol.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
    # # "qm9dft": {
    # #     "mean": [0, 0, 0],
    # #     "std": [1, 1, 1],
    # #     "target_name": ["homo", "lumo", "gap"],
    # # },
    'split_0':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_0':{
        "mean": [242.43846153846152, 79.10384615384615],
        "std": [240.14802178419856, 122.82342385207133],
        "target_name": ['solubility', 'ic50'],
    },
    'hips_solubility_split_0':{
        "mean": [209.10303030303032],
        "std": [180.7239959664737],
        "target_name": ['solubility'],
    },
    'hips_ic50_split_0':{
        "mean": [58.88999999999999],
        "std": [97.6526492216161],
        "target_name": ['ic50'],
    },
    'split_1':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_solubility_split_1':{
        "mean": [206],
        "std": [221],
        "target_name": ['solubility'],
    },
    'hips_ic50_split_1':{
        "mean": [43],
        "std": [68],
        "target_name": ['ic50'],
    },
    'hips_split_1':{
        "mean": [247.37692307692305, 74.8],
        "std": [236.4998834081552, 124.24899937687162],
        "target_name": ['solubility', 'ic50'],
    },
    'split_2':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_2':{
        "mean": [210.55, 56.619230769230775],
        "std": [187.14627403348263, 107.72177488337103],
        "target_name": [],
    },
    'split_3':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_3':{
        "mean": [242.56153846153848, 77.94230769230768],
        "std": [239.00315427098295, 122.94739079121143],
        "target_name": [],
    },
    'hips_split_4':{
        "mean": [185.6153846153846, 38.573076923076925],
        "std": [181.0302560608791, 74.77313666990051],
        "target_name": [],
    },
    'split_4':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'split_5':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_5':{
        "mean": [195.98076923076925, 49.56153846153845],
        "std": [212.25043085800388, 82.34840186634364],
        "target_name": [],
    },
    'split_6':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_6':{
        "mean": [239.98076923076923, 74.13076923076923],
        "std": [267.8854706761982, 121.04180069212825],
        "target_name": [],
    },
    'split_7':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_7':{
        "mean": [235.90769230769232, 56.16153846153847],
        "std": [231.05737702849905, 92.05425271972491],
        "target_name": [],
    },
    'split_8':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_8':{
        "mean": [278.23846153846154, 74.41153846153846],
        "std": [278.0323557221396, 119.14275641148078],
        "target_name": [],
    },
    'split_9':{
        "mean": [],
        "std": [],
        "target_name": [],
    },
    'hips_split_9':{
        "mean": [240.6615384615385, 59.26153846153846],
        "std": [256.09112121833493, 96.05044378595701],
        "target_name": [],
    }
}


@register_task("mol_finetune")
class UniMolFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
        
        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
    
    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad or torch.isnan(loss):
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        
        return loss, sample_size, logging_output
