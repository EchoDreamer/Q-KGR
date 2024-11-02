module load amd/Anaconda/2023.3
source activate graphllm
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0


# dataset_list=("riddle" "obqa" "arc")
# max_new_tokens_list=(10 20 20)
# label_content_type_list=("text_google" "text_google" "text_google_number")

dataset_list=("riddle")
max_new_tokens_list=(10)
label_content_type_list=("text_google")
load_model_path='runs/riddle/greaselm__ds_riddle__enc__k0__sd5__20240608_105259/model.pt'   #riddle
# load_model_path='runs/arc/greaselm__ds_arc__enc__k0__sd5__20240608_111331/model.pt'  #arc

mode='eval_per_class'
model_name='flan_t5_xxl'
eval_batch_size=4

echo "#########evaluate#########"
echo "load_model_path: $load_model_path"
for index in 0; do
    dt=`date '+%Y%m%d_%H%M%S'`
    max_seq_len=200
    dataset=${dataset_list[$index]}
    max_new_tokens=${max_new_tokens_list[$index]}
    label_content_type=${label_content_type_list[$index]}

    if [ ${model_name} = "flan_t5_xl" ]
    then
      encoder='huggingface/model/flan-t5-xl'
      decoder_only=false
    elif [ ${model_name} = "flan_t5_xxl" ]
    then
      encoder='huggingface/model/flan-t5-xxl'
      decoder_only=false
    # elif [ ${model_name} = "llama_7b" ]
    # then
    #   encoder='huggingface/model/Llama-2-7b-hf'
    #   decoder_only=true
    #   max_new_tokens=10
    # elif [ ${model_name} = "llama_7b_chat" ]
    # then
    #   encoder='huggingface/model/Llama-2-7b-chat-hf'
    #   decoder_only=true
    #   max_new_tokens=10
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
    random_ent_emb=false
    echo "***** hyperparameters *****"
    echo "dataset: $dataset"
    echo "max_new_tokens: ${max_new_tokens}"
    echo "label_content_type: ${label_content_type}"
    echo "enc_name: $encoder"
    echo "mode: $mode"
    echo "max_seq_len: $max_seq_len"
    echo "decoder_only: $decoder_only"
    echo "eval_batch_size: $eval_batch_size"
    echo "******************************"
    save_dir_pref='runs'
    mkdir -p $save_dir_pref

    run_name=greaselm__ds_${dataset}__enc__k${k}__sd${seed}__${dt}
    python3 -u q_kgr.py \
        --debug $debug --use_wandb $use_wandb \
        --use_edge_score $use_edge_score\
        --dataset $dataset --max_new_tokens $max_new_tokens --mode $mode \
        --encoder $encoder -k $k  --seed $seed  -sl ${max_seq_len} --max_node_num ${max_node_num} \
        --save_dir ${save_dir_pref}/${dataset}/${run_name} --eval_batch_size $eval_batch_size\
        --run_name ${run_name} --decoder_only $decoder_only --load_model_path $load_model_path\
        --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --ent_emb ${ent_emb//,/ } \
        --data_dir data

    echo "***** hyperparameters *****"
    echo "dataset: $dataset"
    echo "enc_name: $encoder"
    echo "k: $k"
    echo "max_new_tokens: ${max_new_tokens}"
    echo "label_content_type: ${label_content_type}"
    echo "mode: $mode"
    echo "max_seq_len: $max_seq_len"
    echo "decoder_only: $decoder_only"
    echo "eval_batch_size: $eval_batch_size"
    echo "******************************"

done

