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

task_metainfo = {}


@register_task("mol_finetune_ranking_regression")
class UniMolFinetuneRankingRegressionTask(UnicoreTask):
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
            # tgt_dataset = KeyDataset(dataset, "dipole_label")
            tgt_dataset = KeyDataset(dataset, "ranking_label")
            tgt_reg1_dataset = KeyDataset(dataset, "mol1_regression_label")
            tgt_reg2_dataset = KeyDataset(dataset, "mol2_regression_label")
            smi_dataset1 = KeyDataset(dataset, "mol1_smiles")
            smi_dataset2 = KeyDataset(dataset, "mol2_smiles")
            sample1_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "mol1_atom_types", "mol1_atom_coords"
            )
            sample2_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "mol2_atom_types", "mol2_atom_coords"
            )
            dataset1 = AtomTypeDataset(dataset, sample1_dataset)
            dataset2 = AtomTypeDataset(dataset, sample2_dataset)
            
        else:
            # # dataset1 = TTADataset(
            # #     dataset, self.args.seed, "mol1_atom_types", "mol1_atom_coords", self.args.conf_size
            # # )
            # # dataset2 = TTADataset(
            # #     dataset, self.args.seed, "mol2_atom_types", "mol2_atom_coords", self.args.conf_size
            # # )

            sample1_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "mol1_atom_types", "mol1_atom_coords"
            )
            sample2_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "mol2_atom_types", "mol2_atom_coords"
            )

            dataset1 = AtomTypeDataset(dataset, sample1_dataset)
            dataset2 = AtomTypeDataset(dataset, sample2_dataset)

            # tgt_dataset = KeyDataset(dataset, "dipole_label")
            tgt_dataset = KeyDataset(dataset, "ranking_label")
            tgt_reg1_dataset = KeyDataset(dataset, "mol1_regression_label")
            tgt_reg2_dataset = KeyDataset(dataset, "mol2_regression_label")
            smi_dataset1 = KeyDataset(dataset, "mol1_smiles")
            smi_dataset2 = KeyDataset(dataset, "mol2_smiles")

        # tgt_dataset is a list of dict. We want to convert it to a list of list
        tgt_dataset_keys = list(tgt_dataset[0].keys())
        tgt_dataset = [[d[key] for key in tgt_dataset_keys] for d in tgt_dataset]

        dataset1 = CroppingDataset(
            dataset1, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset2 = CroppingDataset(
            dataset2, self.seed, "atoms", "coordinates", self.args.max_atoms
        )

        dataset1 = RemoveHydrogenDataset(
            dataset1,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset2 = RemoveHydrogenDataset(
            dataset2,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset1 = CroppingDataset(
            dataset1, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset2 = CroppingDataset(
            dataset2, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset1 = NormalizeDataset(dataset1, "coordinates", normalize_coord=True)
        dataset2 = NormalizeDataset(dataset2, "coordinates", normalize_coord=True)
        src_dataset1 = KeyDataset(dataset1, "atoms")
        src_dataset2 = KeyDataset(dataset2, "atoms")
        
        src_dataset1 = TokenizeDataset(
            src_dataset1, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        src_dataset2 = TokenizeDataset(
            src_dataset2, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset1 = KeyDataset(dataset1, "coordinates")
        coord_dataset2 = KeyDataset(dataset2, "coordinates")
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset1 = PrependAndAppend(
            src_dataset1, self.dictionary.bos(), self.dictionary.eos()
        )

        src_dataset2 = PrependAndAppend(
            src_dataset2, self.dictionary.bos(), self.dictionary.eos()
        )

        edge_type1 = EdgeTypeDataset(src_dataset1, len(self.dictionary))
        edge_type2 = EdgeTypeDataset(src_dataset2, len(self.dictionary))

        coord_dataset1 = FromNumpyDataset(coord_dataset1)
        coord_dataset2 = FromNumpyDataset(coord_dataset2)


        coord_dataset1 = PrependAndAppend(coord_dataset1, 0.0, 0.0)
        coord_dataset2 = PrependAndAppend(coord_dataset2, 0.0, 0.0)

        distance_dataset1 = DistanceDataset(coord_dataset1)
        distance_dataset2 = DistanceDataset(coord_dataset2)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens1": RightPadDataset(
                        src_dataset1,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_tokens2": RightPadDataset(
                        src_dataset2,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord1": RightPadDatasetCoord(
                        coord_dataset1,
                        pad_idx=0,
                    ),
                    "src_coord2": RightPadDatasetCoord(
                        coord_dataset2,
                        pad_idx=0,
                    ),
                    "src_distance1": RightPadDataset2D(
                        distance_dataset1,
                        pad_idx=0,
                    ),
                    "src_distance2": RightPadDataset2D(
                        distance_dataset2,
                        pad_idx=0,
                    ),
                    "src_edge_type1": RightPadDataset2D(
                        edge_type1,
                        pad_idx=0,
                    ),
                    "src_edge_type2": RightPadDataset2D(
                        edge_type2,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                    'reg1_target': RawLabelDataset(tgt_reg1_dataset),
                    'reg2_target': RawLabelDataset(tgt_reg2_dataset)
                },
                "smi_name1": RawArrayDataset(smi_dataset1),
                "smi_name2": RawArrayDataset(smi_dataset2),
            },
        )
        
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset1))

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
