#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 2-00:00:00
#SBATCH --gres gpu:1
#SBATCH -o step1.out


# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate unimol

data_path=./example_data/gdb17/ # replace to your pretraining data path
save_dir=./save/MolVers/step1 # replace to your save directory
n_gpu=1
MASTER_PORT=10191
lr=1e-4
wd=1e-4
batch_size=32
update_freq=1
masked_token_loss=1.
masked_coord_loss=0.
masked_dist_loss=0.
diffusion_loss=1.
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=-1
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 4 --ddp-backend=c10d \
       --task unimol --loss unimol_drl_noisy_atoms_and_coords --arch unimol_drl_noisy_atoms_and_coords_base  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 100 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar\
       --diffusion-loss $diffusion_loss