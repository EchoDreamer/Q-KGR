#!/bin/bash                              
#SBATCH -p q_intel_gpu_nvidia_nvlink                    
#SBATCH -N 1
#SBATCH -o job_output/job_apex8.out
#SBATCH -e job_output/job_apex8.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
module load amd/Anaconda/2023.3
source activate graphllm
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1
###LLM
decoder_only=false
train_header=false
unfreeze_infuse=true
frozen_lm=false
encoder='huggingface/model/flan-t5-xxl'
###train mode
ddp=false
lora=true
use_trainer=false
mode='train'
elr="1e-4"
gradient_checkpointing=false
dlr="1e-3"
bs=32
mbs=2
seed=5
lr_schedule=fixed
max_epochs_before_stop=10

###dataset
ent_emb=tzw
dataset="obqa"
gnndim=200
max_seq_len=200
max_node_num=200
label_content_type="text_google"
###infuse
infuse_layer="19 20 21 22 23"
num_key_value=5 
k=7   #num of gnn layers

###dege score
use_edge_score=true
context_embedding_zero=true
edge_class='soft'
context_q_a_link_strong=false
question_rel_similarity='BiLinearSimilarity'




###generate
max_new_tokens=20



n_epochs=5





dt=`date '+%Y%m%d_%H%M%S'`
debug=false
use_wandb=false


echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "num_key_value: $num_key_value"
echo "use_edge_score: $use_edge_score"
echo "time: $dt"
echo "******************************"

save_dir_pref='runs'
# mkdir -p $save_dir_pref

run_name=greaselm__ds_${dataset}__enc__k${k}__sd${seed}__${dt}
log=logs/${dataset}/${train}_${run_name}.log.txt
# log=logs/${dataset}/$train_${run_name}.log.txt
###### Training ######
python q_kgr.py \
     --debug $debug --use_wandb $use_wandb --context_q_a_link_strong $context_q_a_link_strong --gradient_checkpointing $gradient_checkpointing\
     --use_edge_score $use_edge_score  --edge_class $edge_class --question_rel_similarity $question_rel_similarity\
    --dataset $dataset --max_new_tokens $max_new_tokens --frozen_lm $frozen_lm --infuse_layer $infuse_layer --use_trainer $use_trainer\
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --unfreeze_infuse $unfreeze_infuse\
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --train_header $train_header --mode $mode --ddp $ddp \
    --run_name ${run_name} --context_embedding_zero $context_embedding_zero --decoder_only $decoder_only --label_content_type $label_content_type\
    --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule} --lora $lora\
    --data_dir data --num_key_value $num_key_value \
> ${log}
echo log: $log

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "num_key_value: $num_key_value"
echo "use_edge_score: $use_edge_score"
echo "time: $dt"
echo "******************************"