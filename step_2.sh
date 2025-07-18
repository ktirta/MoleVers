#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -o step2.out


# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate unimol

data_path="./example_data/gdb17_step2"  # replace to your pretraining data path
save_dir="./save/MolVers/step2"  # replace to your save path
n_gpu=1
MASTER_PORT=10074
dict_name="dict.txt"
weight_path="./save/MolVers/step1/checkpoint_last.pt"  # replace to your ckpt path
task_name="qm9dft"  # molecular property prediction task name 
task_num=3
loss_func="finetune_smooth_mae"
lr=1e-4
batch_size=32
epoch=50
dropout=0.2
warmup=0.02
local_batch_size=32
only_polar=-1
conf_size=1
seed=0

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -u -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --conf-size $conf_size \
       --num-workers 4 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_drl_noisy_atoms_and_coords_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 50 \
       --save-dir $save_dir --only-polar $only_polar \
       --tmp-save-dir $save_dir/tmp