#!/bin/bash                              
#SBATCH -p q_intel_gpu_nvidia_nvlink                    
#SBATCH -N 1
#SBATCH -o job_apex.out
#SBATCH -e job_apex.err
#SBATCH --gres=gpu:1
export CUDA_VISIBLE_DEVICES=0
hostname 
module load amd/Anaconda/2023.3
source activate graphllm
export TOKENIZERS_PARALLELISM=true
stage1_train=true
decoder_only=false
ddp=false
use_trainer=false

mode='train'
dt=`date '+%Y%m%d_%H%M%S'`
use_edge_score=true
context_embedding_zero=true
elr="1e-4"
train_header=false
gradient_checkpointing=false

infuse_layer="2 3 4 5 6"
unfreeze_infuse=true

edge_class='soft'
context_q_a_link_strong=false
question_rel_similarity='BiLinearSimilarity'
debug=false
use_wandb=false
frozen_lm=false
num_key_value=5   
k=7  #num of gnn layers
dataset="obqa"
shift
# encoder='huggingface/llama2-chat-7B'
encoder='huggingface/model/flan-t5-xxl'
# encoder='huggingface/model/Meta-Llama-3-8B-Instruct'

args=$@
max_new_tokens=20


dlr="1e-3"
bs=16
mbs=2

gnndim=200
max_seq_len=200
# Existing arguments but changed for GreaseLM
encoder_layer=-1
max_node_num=200
seed=5
lr_schedule=fixed
if [ ${dataset} = "obqa" ];
then
  max_seq_len=200
else
  max_seq_len=100
fi

if [ ${stage1_train} = true ];
then
  n_epochs=5
  frozen_lm=true
  unfreeze_infuse=true
  train_header=false
  echo "stage1 training"
else
  n_epochs=3
fi
max_epochs_before_stop=10




ent_emb=tzw

# Added for GreaseLM
resume_checkpoint=None
resume_id=None

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
log=logs/train_${dataset}__${run_name}.log.txt
# log=logs/${dataset}/$train_${run_name}.log.txt
###### Training ######
python q_kgr.py \
     --debug $debug --use_wandb $use_wandb --context_q_a_link_strong $context_q_a_link_strong --gradient_checkpointing $gradient_checkpointing\
     --use_edge_score $use_edge_score  --edge_class $edge_class --question_rel_similarity $question_rel_similarity\
    --dataset $dataset --max_new_tokens $max_new_tokens --frozen_lm $frozen_lm --infuse_layer $infuse_layer --use_trainer $use_trainer\
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs}  --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --unfreeze_infuse $unfreeze_infuse\
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --train_header $train_header --mode $mode --ddp $ddp\
    --run_name ${run_name} --context_embedding_zero $context_embedding_zero --decoder_only $decoder_only\
    --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule} \
    --data_dir data --num_key_value $num_key_value \
    $args \
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
echo "******************************"