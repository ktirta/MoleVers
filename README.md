## Two-Stage Pretraining for Molecular Property Prediction in the Wild

Source code for Two-Stage Pretraining for Molecular Property Prediction in the Wild

Our code is based on [Uni-Mol](https://github.com/deepmodeling/Uni-Mol). 
Please clone their repository, install their dependencies ([Uni-Core](https://github.com/dptech-corp/Uni-Core#installation), rdkit), and replace the directory `./Uni-Mol/unimol/unimol´ with our version of unimol.

We use `slurm` to run the training and evaluation scripts.

To start the first stage pretraining, run `step1.sh`.

To start the second stage pretraining, run `step2.sh`.

To finetune and evaluate on downstream data, run `step3.sh`

The complete evaluation datasets and pretrained model will be released soon.
