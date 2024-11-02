module load amd/Anaconda/2023.3
source activate graphllm
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=3
checkpoint_base="/home/export/base/ycsc_chenkh/hitici_03/online1/final_version/checkpoint_emnlp_v1/main"

#test obqa
# dataset="obqa"
# max_new_tokens=20
# label_content_type="text_google"

# test riddle
# dataset="riddle"
# max_new_tokens=10
# label_content_type="text_google"

#test arc
dataset=arc
max_new_tokens=20
label_content_type="text_google_number"

# test piqa
# dataset="piqa"
# max_new_tokens=10
# label_content_type="text_google_number"

load_model_path='$checkpoint_base/model.pt'

merge_lora_test=false
mode='eval_lora'
# mode='eval_lora'
model_name='flan_t5_xl'
max_seq_len=200
eval_batch_size=4
echo "#########evaluate_lora#########"
echo "load_model_path: $load_model_path"

dt=`date '+%Y%m%d_%H%M%S'`
if [ ${model_name} = "flan_t5_xl" ]
then
  encoder='huggingface/model/flan-t5-xl'
  decoder_only=false
elif [ ${model_name} = "flan_t5_xxl" ]
then
  encoder='huggingface/model/flan-t5-xxl'
  decoder_only=false
else
  echo "model not found"
  exit
fi
k=2 #num of gnn layers
ent_emb=tzw
debug=false
use_wandb=false
max_node_num=200
seed=5
use_edge_score=false
resume_checkpoint=None
resume_id=None
echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "max_new_tokens: ${max_new_tokens}"
echo "label_content_type: ${label_content_type}"
echo "enc_name: $encoder"
echo "mode: $mode"
echo "max_seq_len: $max_seq_len"
echo "decoder_only: $decoder_only"
echo "eval_batch_size: $eval_batch_size"
echo "merge_lora_test: $merge_lora_test"
echo "******************************"
save_dir_pref='runs'
mkdir -p $save_dir_pref
run_name=greaselm__ds_${dataset}__enc__k${k}__sd${seed}__${dt}
python3 -u q_kgr.py \
  --debug $debug --use_wandb $use_wandb \
  --use_edge_score $use_edge_score --label_content_type $label_content_type --merge_lora_test $merge_lora_test\
  --dataset $dataset --max_new_tokens $max_new_tokens --mode $mode \
  --encoder $encoder -k $k  --seed $seed  -sl ${max_seq_len} --max_node_num ${max_node_num} \
  --save_dir ${save_dir_pref}/${dataset}/${run_name} --eval_batch_size $eval_batch_size \
  --run_name ${run_name} --decoder_only $decoder_only --load_model_path $load_model_path\
  --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --ent_emb ${ent_emb//,/ } \
  --data_dir data 

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "max_new_tokens: ${max_new_tokens}"
echo "label_content_type: ${label_content_type}"
echo "enc_name: $encoder"
echo "mode: $mode"
echo "max_seq_len: $max_seq_len"
echo "decoder_only: $decoder_only"
echo "eval_batch_size: $eval_batch_size"
echo "merge_lora_test: $merge_lora_test"
echo "******************************"
