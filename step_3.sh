#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 00:10:00
#SBATCH --gres gpu:1
#SBATCH -o step3.out


# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate unimol

split_idx=0 # choose the split id of the dataset {0, 1, 2}
MASTER_PORT="$(($split_idx + 10045))"

data_path=./example_data/ChemBL_assays/assay1_CHEMBL635482_logp  # replace to your downstream data path
save_dir=./save/MolVers/step3/assay1_$split_idx  # replace to your save path

n_gpu=1
dict_name="dict.txt"
weight_path="./save/MolVers/step2/checkpoint_last.pt"  # replace to your ckpt path
task_name=split_$split_idx  # molecular property prediction task name 
task_num=1
loss_func="finetune_smooth_mae"
lr=1e-4
batch_size=32
epoch=50
dropout=0
warmup=0.06
local_batch_size=32
only_polar=-1
conf_size=1
seed=1

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "split_0" ] || [ "$task_name" == "split_1" ] || [ "$task_name" == "split_2" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -u -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name \
       --user-dir ./unimol --train-subset train --valid-subset valid \
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
       --tmp-save-dir $save_dir/tmp  --validate-interval 1 --save-interval 50
       