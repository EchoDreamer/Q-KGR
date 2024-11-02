import datetime
time_now= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
import os

import argparse
import logging as python_log
import random
import shutil
import time
import copy
from peft import LoraConfig,get_peft_model,TaskType
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb
from modeling import modeling_greaselm
from utils import data_utils
from utils import optimization_utils
from utils import parser_utils
from utils import utils


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DECODER_DEFAULT_LR = {
    'obqa': 3e-4,
    'riddle': 3e-3,     
    'arc':3e-3,
    'piqa':3e-3

 }

import numpy as np

import socket, os, subprocess

logger = python_log.getLogger(__name__)


def load_data(args, devices, kg):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)


    dataset = data_utils.GreaseLM_DataLoader(args.train_statements, args.train_adj,
        args.dev_statements, args.dev_adj,
        args.test_statements, args.test_adj,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all,
        kg=kg,lm_label_type=args.lm_label_type,dataset_name=args.dataset,ddp=args.ddp,label_content_type=args.label_content_type,
        num_context=args.num_context)

    return dataset


def construct_model(args, kg,lora_config=None,lora_test_dir=None):

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    if args.decoder_only:
        model = modeling_greaselm.GreaseLM(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
            concept_dim=args.gnn_dim,
            concept_in_dim=concept_in_dim,
            n_attention_head=args.att_head_num,
            p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
            pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
            init_range=args.init_range,
            infuse_layer=args.infuse_layer,num_key_value=args.num_key_value,
            use_edge_score=args.use_edge_score,question_rel_similarity=args.question_rel_similarity,
            edge_class=args.edge_class,context_q_a_link_strong=args.context_q_a_link_strong,context_embedding_zero=args.context_embedding_zero,
            unfreeze_infuse=args.unfreeze_infuse,train_header=args.train_header,gradient_checkpointing=args.gradient_checkpointing,
            max_new_tokens=args.max_new_tokens,lora_config=lora_config)
    
    else:
        model = modeling_greaselm.GreaseLM_(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
            concept_dim=args.gnn_dim,
            concept_in_dim=concept_in_dim,
            n_attention_head=args.att_head_num,
            p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
            pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
            init_range=args.init_range,
            infuse_layer=args.infuse_layer,num_key_value=args.num_key_value,
            use_edge_score=args.use_edge_score,question_rel_similarity=args.question_rel_similarity,
            edge_class=args.edge_class,context_q_a_link_strong=args.context_q_a_link_strong,context_embedding_zero=args.context_embedding_zero,
            unfreeze_infuse=args.unfreeze_infuse,train_header=args.train_header,gradient_checkpointing=args.gradient_checkpointing,
            max_new_tokens=args.max_new_tokens,lora_config=lora_config,lora_test_dir=lora_test_dir,
            merge_lora_test=args.merge_lora_test)
    
    return model
def sep_params_keys(model, loaded_llama_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()    
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = []
    large_lr_params = []
    for n, p in model.named_parameters():
        if n in loaded_llama_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params.append("_fsdp_wrapped_module."+n)
        else:
            not_loaded_params[n] = p
            large_lr_params.append("_fsdp_wrapped_module."+n)

    return loaded_params, not_loaded_params, small_lr_params, large_lr_params

def sep_params(model, loaded_llama_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()    
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_llama_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, small_lr_params, large_lr_params
def count_parameters(loaded_params, not_loaded_params):
    num_train_not_loaded_params = sum(p.numel() for p in not_loaded_params.values() if p.requires_grad)   
    num_fixed_not_loaded_params = sum(p.numel() for p in not_loaded_params.values() if not p.requires_grad)   
    num_train_loaded_params = sum(p.numel() for p in loaded_params.values() if p.requires_grad)  
    num_fixed_loaded_params = sum(p.numel() for p in loaded_params.values() if not p.requires_grad)
    

    
    
    print('num_loaded_trainable_params:', num_train_loaded_params)
    print('num_not_loaded_trainable_params:', num_train_not_loaded_params)
    print('num_loaded_fixed_params:', num_fixed_loaded_params)
    print('num_not_loaded_fixed_params:', num_fixed_not_loaded_params)
    print('num_llm_params:', num_train_loaded_params+num_fixed_loaded_params)
    print('num_all_params:',num_train_not_loaded_params+num_fixed_not_loaded_params+num_fixed_loaded_params+num_train_loaded_params)

def calc_loss_and_acc(logits, loss_type, loss_func, *labels):
    label=labels[0]
    bs = label.size(0)

    if loss_type == 'margin_rank':
        num_choice = logits.size(1)
        flat_logits = logits.view(-1)
        correct_mask = F.one_hot(label.view(-1), num_classes=num_choice).view(-1)  # of length batch_size*num_choice
        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
        wrong_logits = flat_logits[correct_mask == 0]
        y = wrong_logits.new_ones((wrong_logits.size(0),))
        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
    elif loss_type == 'cross_entropy':
        loss = loss_func(logits,label.view(-1))
    loss *= bs


    return loss
def calc_eval_accuracy(eval_set, model, debug,tokenizer,log_writer=None,max_new_tokens=5,test_mode='online'):
    """Eval on the dev or test set - calculate loss and accuracy"""
    model.eval()
    with torch.no_grad():
        count_correct=0
        count_sum=0
        for qids, *input_data in tqdm(eval_set, desc="Dev/Test batch"):
            # bs = labels[0].size(0)
            bs=input_data[0].size(0)
            labels=input_data[0]
            if test_mode=='online': 
                replaced_tensor = torch.where(labels == -100, torch.tensor(1), labels).view(bs, -1)
                token_label=tokenizer.batch_decode(replaced_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                token_label=[]
            # if token_label[0]=='The size of a statue':
            #     print(token_label)
            result = model.generate(*input_data,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1)  
            if model.lmgnn.model_type=='llama':
                token_result=tokenizer.batch_decode(result[:,input_data[1].size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:   
                token_result=tokenizer.batch_decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            count_sum+=bs
            
            if test_mode=='online':
                log_writer.write('ground_label,pred_label\n')
                for sub_label,sub_result in zip (token_label,token_result):
                
                    if model.lmgnn.model_type=='T5':
                        if args.label_content_type=='text_google':
                            if sub_label.lower() == sub_result.lower() or sub_label.lower() == sub_result.lower()+".":
                                count_correct+=1
                            else:
                                if log_writer is not None:
                                    log_writer.write('{},{}\n'.format(sub_label,sub_result ))   
                        elif args.label_content_type=='text_google_number':
                            if len(sub_label)>0 and len(sub_result)>0 and sub_label[0] == sub_result[0]:
                                count_correct+=1
                            else:
                                if log_writer is not None:
                                    log_writer.write('{},{}\n'.format(sub_label,sub_result ))   
                    elif model.lmgnn.model_type=='llama':
                        if sub_label.lower() in sub_result.lower():
                            count_correct+=1
                        else:
                            if log_writer is not None:
                                try:
                                    log_writer.write('{},{}\n'.format(sub_label,sub_result ))
                                except:
                                    log_writer.write('{},UnicodeEncodeError\n'.format(sub_label ))
            elif test_mode=='offline':
                for sub_result in token_result:
                    if model.lmgnn.model_type=='T5':
                        if args.label_content_type=='text_google':
                            try:
                                log_writer.write('{}\n'.format(sub_result))
                            except:
                                log_writer.write('0\n')
                        elif args.label_content_type=='text_google_number':
                            if len(sub_result)>0:
                                try:
                                    log_writer.write('{}\n'.format(sub_result[0]))   
                                except:
                                    log_writer.write('0\n')
                            else:
                                log_writer.write('0\n')
                    elif model.lmgnn.model_type=='llama':
                        try:
                            log_writer.write('{}\n'.format(sub_result))
                        except:
                            log_writer.write('0\n')
            else:
                raise ValueError("Invalid test_mode.")
                
    return count_correct/count_sum



def evaluate_zero_shot(args, has_test_split, devices, kg):
    print("args: {}".format(args))

    if not args.debug:
        config_path = os.path.join(args.save_dir, 'config.json')
        utils.export_config(args, config_path)
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg)
    # model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))
    
    
    if model.lmgnn.model_type=='llama':
        para_llm=["lmgnn.mp.model."+name for name,_ in model.lmgnn.mp.model.named_parameters()]
        para_lm_head=["lmgnn.mp.lm_head."+name for name,_ in model.lmgnn.mp.lm_head.named_parameters()]
        loaded_llm_keys=para_llm+para_lm_head
    elif model.lmgnn.model_type=='T5':
        para_encoder=["lmgnn.mp.encoder."+name for name,_ in model.lmgnn.mp.encoder.named_parameters() if name!= 'embed_tokens.weight']
        para_decoder=["lmgnn.mp.decoder."+name for name,_ in model.lmgnn.mp.decoder.named_parameters() if name!= 'embed_tokens.weight']
        para_lm_head=["lmgnn.mp.lm_head."+name for name,_ in model.lmgnn.mp.lm_head.named_parameters()]
        para_shared=["lmgnn.mp.shared."+name for name,_ in model.lmgnn.mp.shared.named_parameters()]
        loaded_llm_keys=para_encoder+para_decoder+para_lm_head+para_shared
    else:
        raise ValueError("Invalid model type.")
    # Separate the parameters into loaded and not loaded
    loaded_params, not_loaded_params, small_lr_params, large_lr_params = sep_params(model, loaded_llm_keys)
    loaded_params, not_loaded_params,  small_lr_params_keys, large_lr_params_keys=sep_params_keys(model, loaded_llm_keys)
    # print non-loaded parameters
    print('Non-loaded parameters:')
    for name, param in not_loaded_params.items():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

    # Count parameters
    count_parameters(loaded_params, not_loaded_params)
    

    model.to(devices[1])
    if args.k>0:
        if args.context_embedding_zero:
            model.lmgnn.concept_emb.to(devices[0])
        else:
            model.lmgnn.mp.concept_emb.to(devices[0])
    if True:

        model.eval()
        dev_acc=0
        preds_path_dev = os.path.join(args.save_dir, 'dev_preds.csv')
        with open(preds_path_dev, 'w') as fout:
            # fout.write('ground_label,pred_label\n')
            dev_acc = calc_eval_accuracy(dev_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,max_new_tokens=args.max_new_tokens)

        if has_test_split:
            preds_path_test = os.path.join(args.save_dir, 'test_preds.csv')
            with open(preds_path_test, 'w') as fout:


                test_acc = calc_eval_accuracy(test_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,max_new_tokens=args.max_new_tokens,
                                              test_mode=args.test_mode)
        else:
            test_acc = 0
        print('| dev_acc {:7.4f} | test_acc {:7.4f} |'.format(dev_acc, test_acc))

def evaluate(args, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse
    save_dir=args.save_dir
    label_content_type=args.label_content_type
    max_new_tokens=args.max_new_tokens
    eval_batch_size=args.eval_batch_size
    #需要改一下
    args = utils.import_config(checkpoint["config"], args)
    
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse
    args.save_dir=save_dir
    args.label_content_type=label_content_type
    args.max_new_tokens=max_new_tokens
    args.eval_batch_size=eval_batch_size
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg)

    xxx=checkpoint["model"]
    if "lmgnn.concept_emb.emb.weight" in xxx:
        del xxx["lmgnn.concept_emb.emb.weight"]
    model.load_state_dict(xxx, strict=False)
    epoch_id = checkpoint['epoch']

    model.to(devices[1])
    if args.k>0:
        if args.context_embedding_zero:
            model.lmgnn.concept_emb.to(devices[0])
        else:
            model.lmgnn.mp.concept_emb.to(devices[0])


    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)


    if True:
        model.eval()
        dev_acc=0
        preds_path_dev = os.path.join(args.save_dir, 'dev_preds.csv')
        utils.check_path(preds_path_dev)
            
        with open(preds_path_dev, 'w') as fout:
            # fout.write('ground_label,pred_label\n')
            dev_acc = calc_eval_accuracy(dev_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                         max_new_tokens=args.max_new_tokens)

        if has_test_split:
            preds_path_test = os.path.join(args.save_dir, 'test_preds.csv')
            with open(preds_path_test, 'w') as fout:
                # fout.write('ground_label,pred_label\n')
                test_acc = calc_eval_accuracy(test_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                              max_new_tokens=args.max_new_tokens,test_mode=args.test_mode)
        else:
            test_acc = 0
        print('| dev_acc {:7.4f} | test_acc {:7.4f} |'.format(dev_acc, test_acc))

def evaluate_lora(args, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    max_new_tokens=args.max_new_tokens
    debug = args.debug
    inhouse = args.inhouse
    save_dir=args.save_dir
    label_content_type=args.label_content_type
    args = utils.import_config(checkpoint["config"], args)
    args.max_new_tokens=max_new_tokens
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse
    args.save_dir=save_dir
    args.label_content_type=label_content_type
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    load_model_path_lora = os.path.dirname(load_model_path)
    if os.path.exists(os.path.join(load_model_path_lora,'adapter_model.safetensors')):
        print("load lora checkpoint\n")
        model = construct_model(args, kg,lora_test_dir=load_model_path_lora)
    else:

        model = construct_model(args, kg)
    # model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))
    xxx=checkpoint["model"]
    if "lmgnn.concept_emb.emb.weight" in xxx:
        del xxx["lmgnn.concept_emb.emb.weight"]
    model.load_state_dict(xxx, strict=False)
    epoch_id = checkpoint['epoch']

    model.to(devices[1])
    if args.k>0:
        if args.context_embedding_zero:
            model.lmgnn.concept_emb.to(devices[0])
        else:
            model.lmgnn.mp.concept_emb.to(devices[0])


    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)


    if True:
        model.eval()
        dev_acc=0
        preds_path_dev = os.path.join(args.save_dir, 'dev_preds.csv')
        utils.check_path(preds_path_dev)
            
        with open(preds_path_dev, 'w') as fout:
            # fout.write('ground_label,pred_label\n')
            dev_acc = calc_eval_accuracy(dev_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                         max_new_tokens=args.max_new_tokens)

        if has_test_split:
            preds_path_test = os.path.join(args.save_dir, 'test_preds.csv')
            with open(preds_path_test, 'w') as fout:
                # fout.write('ground_label,pred_label\n')
                test_acc = calc_eval_accuracy(test_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                              max_new_tokens=args.max_new_tokens,test_mode=args.test_mode)
        else:
            test_acc = 0
        print('| dev_acc {:7.4f} | test_acc {:7.4f} |'.format(dev_acc, test_acc))



def train(args, resume, has_test_split, devices, kg):
    print("args: {}".format(args))
    if args.lora:
        if args.decoder_only:
            lora_target_modules=['gate_proj','up_proj','down_proj']
            task_type=TaskType.CAUSAL_LM
        else:
            lora_target_modules=['wi_0','wi_1','wo']
            task_type=TaskType.SEQ_2_SEQ_LM
        lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=task_type,
    )
    else:
        lora_config=None
        
        
    if resume:
        args.save_dir = os.path.dirname(args.resume_checkpoint)
        print(args.save_dir)
    if not args.debug and not args.use_trainer:
        log_path = os.path.join(args.save_dir, 'log.csv')
        utils.check_path(log_path)
        # Set up tensorboard
        # tb_dir=args.save_dir
        tb_dir = os.path.join(args.save_dir, "tb")
        if not resume:
            with open(log_path, 'w') as fout:
                fout.write('epoch,step,dev_acc,test_acc,best_dev_acc,final_test_acc,best_dev_epoch\n')

            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

        config_path = os.path.join(args.save_dir, 'config.json')
        utils.export_config(args, config_path)
        model_path = os.path.join(args.save_dir, 'model.pt')
    else:
        config_path = ('./output/config.json')
        utils.export_config(args, config_path)
    dataset = load_data(args, devices, kg)
    if args.use_trainer:
        train_dataloader = dataset.train_dataset()
        dev_dataloader = dataset.dev_dataset()
        data_collator = data_utils.DataCollatorForDreamLLMDataset()
    else:
        train_dataloader = dataset.train()
        dev_dataloader = dataset.dev()
    
    def _rename_key(key):
        if key.startswith("model."):
            return key.replace("model.", "lmgnn.mp.")
        else:
            return "lmgnn.mp." + key
    if has_test_split:
        test_dataloader = dataset.test()
        # test_dataloader = dataset.test_dataset()
    model = construct_model(args, kg,lora_config)
    # model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))
    # for k,v in model.named_parameters():
    #     print(k)

    if model.lmgnn.model_type=='llama':
        para_llm=["lmgnn.mp.model."+name for name,_ in model.lmgnn.mp.model.named_parameters()]
        para_lm_head=["lmgnn.mp.lm_head."+name for name,_ in model.lmgnn.mp.lm_head.named_parameters()]
        loaded_llm_keys=para_llm+para_lm_head
    elif model.lmgnn.model_type=='T5':
        para_encoder=["lmgnn.mp.encoder."+name for name,_ in model.lmgnn.mp.encoder.named_parameters() if name!= 'embed_tokens.weight']
        para_decoder=["lmgnn.mp.decoder."+name for name,_ in model.lmgnn.mp.decoder.named_parameters() if name!= 'embed_tokens.weight']
        para_lm_head=["lmgnn.mp.lm_head."+name for name,_ in model.lmgnn.mp.lm_head.named_parameters()]
        para_shared=["lmgnn.mp.shared."+name for name,_ in model.lmgnn.mp.shared.named_parameters()]
        loaded_llm_keys=para_encoder+para_decoder+para_lm_head+para_shared
    else:
        raise ValueError("Invalid model type.")
    # Separate the parameters into loaded and not loaded
    loaded_params, not_loaded_params, small_lr_params, large_lr_params = sep_params(model, loaded_llm_keys)
    loaded_params, not_loaded_params,  small_lr_params_keys, large_lr_params_keys=sep_params_keys(model, loaded_llm_keys)
    # print non-loaded parameters
    print('Non-loaded parameters:')
    for name, param in not_loaded_params.items():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

    # Count parameters
    count_parameters(loaded_params, not_loaded_params)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


    # Create an optimizer
    grouped_parameters = [
        {'params': [p for n, p in small_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in small_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in large_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]

    optimizer = optimization_utils.OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    
    
    
    
    # optimizer = optimization_utils.OPTIMIZER_CLASSES[args.optim](model.parameters(), lr=args.decoder_lr)

    #########################################################
    # Optionally loading from a checkpoint
    #########################################################
    # checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), 
    #               "epoch": epoch_id, "global_step": global_step, 
    #               "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
    if resume:
        print("loading from checkpoint: {}".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        last_epoch = -1
        global_step = 0
        model.load_state_dict(checkpoint["model"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        final_test_acc = checkpoint["final_test_acc"]
        best_dev_epoch = 0
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = final_test_acc = 0


    #########################################################
    # Create a scheduler
    #########################################################
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, last_epoch=last_epoch)
    
    if resume:
        scheduler.load_state_dict(checkpoint["scheduler"])
    

    if not args.use_trainer:
        model.to(devices[1])
        if args.k>0:
            if  args.context_embedding_zero:
                model.lmgnn.concept_emb.to(devices[0])
            else:
                model.lmgnn.mp.concept_emb.to(devices[0])

    
    
    if model.lmgnn.model_type=='llama':
        fsdp_config={
            "transformer_layer_cls_to_wrap":"LlamaDecoderLayer",
            "use_orig_params":"True",
            "ignored_params":[param for param in model.parameters() if not param.requires_grad]
        }
    elif model.lmgnn.model_type=='T5':
        fsdp_config={
            "transformer_layer_cls_to_wrap":"T5Block",
            "use_orig_params":"True",
            "ignored_params":[param for param in model.parameters() if not param.requires_grad]
        }
    
    if args.use_trainer:
    
        trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataloader,
        eval_dataset=dev_dataloader,
        # compute_metrics=data_utils.ComputeMetrics(dataset.tokenizer),
        # optimizers=(optimizer, scheduler),
        
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.mini_batch_size,    
            gradient_accumulation_steps=args.batch_size//args.mini_batch_size,   
            num_train_epochs=args.n_epochs,
            learning_rate=args.encoder_lr,
            # fp16=True,
            fp16=False,
            logging_steps=100,   
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-5,
            prediction_loss_only=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            weight_decay=args.weight_decay,
            # evaluation_strategy='steps',
            # eval_steps=100, #if val_set_size > 0 else None,
            # save_steps=100,
            # label_smoother=0.5,    
            output_dir='./output',
            save_total_limit=4,   
            load_best_model_at_end=True, #if val_set_size > 0 else False,
            ddp_find_unused_parameters=False,#if ddp else None,
            group_by_length=False,
            report_to="wandb" if args.use_wandb else None,
            run_name=None, #wandb_run_name if use_wandb else None,
            fsdp ="full_shard auto_wrap",
            fsdp_config=fsdp_config,
            gradient_checkpointing=True
            # fsdp_transformer_layer_cls_to_wrap ="LlamaDecoderLayer" 
        ),
        data_collator=data_collator
    )
        trainer.train()
        torch.save(args,'./output/args_ours.pt')     
        
    else:
        model.train()
        print()
        print('-' * 71)

        total_loss_acm = 0.0
        n_samples_acm = n_corrects_acm = 0
        total_time = 0
        for epoch_id in trange(last_epoch + 1, args.n_epochs, desc="Epoch"):
            model.train()
            # lora_test_num=0
            for qids, *input_data in tqdm(train_dataloader, desc="Batch"):

                start_time = time.time()
                optimizer.zero_grad()
                bs = len(qids)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    
                    output= model(*[x[a:b] for x in input_data])
                    logits=output['logits']
                    loss=output['loss']

                    total_loss_acm += loss.item()
                    loss = loss / bs
                    loss.backward()
                    # n_corrects_acm += n_corrects
                    n_samples_acm += (b - a)

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Gradients are accumulated and not back-proped until a batch is processed (not a mini-batch).
                optimizer.step()
                scheduler.step()
                total_time += (time.time() - start_time)

                if (global_step + 1) % args.log_interval == 0:
                    ms_per_batch = 1000 * total_time / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))

                    if not args.debug:
                        tb_writer.add_scalar('Train/Lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('Train/Loss', total_loss_acm / n_samples_acm, global_step)
                        # tb_writer.add_scalar('Train/Acc', n_corrects_acm / n_samples_acm, global_step)
                        tb_writer.add_scalar('Train/ms_per_batch', ms_per_batch, global_step)
                        tb_writer.flush()
                    wandb.log({"lr": scheduler.get_lr()[0], "train_loss": total_loss_acm / n_samples_acm, "train_acc": n_corrects_acm / n_samples_acm, "ms_per_batch": ms_per_batch}, step=global_step)

                    total_loss_acm = 0.0
                    n_samples_acm = n_corrects_acm = 0
                    total_time = 0
                global_step += 1 # Number of batches processed up to now
            
            
            # file_logit.close()
            # Save checkpoints and evaluate after every epoch
            


            model.eval()
            dev_acc=0
            preds_path_dev = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
            with open(preds_path_dev, 'w') as fout:
                # fout.write('ground_label,pred_label\n')
                dev_acc = calc_eval_accuracy(dev_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,max_new_tokens=args.max_new_tokens)

            if has_test_split:
                preds_path_test = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                with open(preds_path_test, 'w') as fout:
                    # fout.write('ground_label,pred_label\n')
                    test_acc = calc_eval_accuracy(test_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                                  max_new_tokens=args.max_new_tokens,test_mode=args.test_mode)
            else:
                test_acc = 0
            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)

            if not args.debug:
                tb_writer.add_scalar('Dev/Acc', dev_acc, global_step)
                if has_test_split:
                    tb_writer.add_scalar('Test/Acc', test_acc, global_step)
                tb_writer.flush()
                
                
                
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    copy_model_lora=False
                    if args.lora:
                        if copy_model_lora:
                            model.to('cpu')
                            model_copy=copy.deepcopy(model)
                            if not args.use_trainer:
                                model.to(devices[1])
                                if args.k>0:
                                    if  args.context_embedding_zero:
                                        model.lmgnn.concept_emb.to(devices[0])
                                    else:
                                        model.lmgnn.mp.concept_emb.to(devices[0])
                            model_copy.lmgnn.mp=model_copy.lmgnn.mp.merge_and_unload()
                            model_state_dict = model_copy.state_dict()
                            if args.k>0:
                                del model_state_dict["lmgnn.concept_emb.emb.weight"]
                            checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
                            print('Saving model to {}.{}'.format(model_path, epoch_id))
                            torch.save(checkpoint, model_path)
                        else:
                            
                            model.lmgnn.mp.save_pretrained(args.save_dir)   #save Lora
                            model_state_dict = model.state_dict()    
                            if args.k>0:
                                del model_state_dict["lmgnn.concept_emb.emb.weight"]
                            for llm_model_name,_ in model.named_parameters():
                                if 'encoder' in llm_model_name or 'decoder' in llm_model_name or 'lm_head' in llm_model_name or 'shared' in llm_model_name:
                                    del model_state_dict[llm_model_name]    
                            checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
                            print('Saving model to {}.{}'.format(model_path, epoch_id))
                            torch.save(checkpoint, model_path)
                    else:
                        model_state_dict = model.state_dict()
                        if args.k>0:
                            del model_state_dict["lmgnn.concept_emb.emb.weight"]
                        checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
                        print('Saving model to {}.{}'.format(model_path, epoch_id))
                        torch.save(checkpoint, model_path)
                        
                        
            if not args.debug:
                with open(log_path, 'a') as fout:
                    fout.write('{:3},{:5},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:3}\n'.format(epoch_id, global_step, dev_acc, test_acc, best_dev_acc, final_test_acc, best_dev_epoch))

            wandb.log({"dev_acc": dev_acc, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
            if has_test_split:
                wandb.log({"test_acc": test_acc, "final_test_acc": final_test_acc}, step=global_step)

            model.train()
            start_time = time.time()
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break

            if args.debug:
                break

        if not args.debug: 
            tb_writer.close()



def evaluate_ddp(args, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_state_dict = args.load_model_state_dict
    load_train_args=args.load_train_args
    print("loading from checkpoint: {}, train_args: {}".format(load_model_state_dict,load_train_args))
    checkpoint_model = torch.load(load_model_state_dict, map_location='cpu')
    checkpoint_args=torch.load(load_train_args, map_location='cpu')
    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse
    save_dir=args.save_dir
    
    args = utils.import_config(checkpoint_args, args)
    
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse
    args.save_dir=save_dir
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg)


    # if "lmgnn.concept_emb.emb.weight" in checkpoint_model:
    #     del checkpoint_model["lmgnn.concept_emb.emb.weight"]
    model.load_state_dict(checkpoint_model, strict=False)


    model.to(devices[1])
    if args.k>0:
        if args.context_embedding_zero:
            model.lmgnn.concept_emb.to(devices[0])
        else:
            model.lmgnn.mp.concept_emb.to(devices[0])

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)


    if True:
        model.eval()
        dev_acc=0
        preds_path_dev = os.path.join(args.save_dir, 'dev_preds.csv')
        utils.check_path(preds_path_dev)
            
        with open(preds_path_dev, 'w') as fout:
            # fout.write('ground_label,pred_label\n')
            dev_acc = calc_eval_accuracy(dev_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                         max_new_tokens=args.max_new_tokens)

        if has_test_split:
            preds_path_test = os.path.join(args.save_dir, 'test_preds.csv')
            with open(preds_path_test, 'w') as fout:
                # fout.write('ground_label,pred_label\n')
                test_acc = calc_eval_accuracy(test_dataloader, model, args.debug,tokenizer=dataset.tokenizer,log_writer=fout,
                                              max_new_tokens=args.max_new_tokens,test_mode=args.test_mode)
        else:
            test_acc = 0
        print('| dev_acc {:7.4f} | test_acc {:7.4f} |'.format(dev_acc, test_acc))


def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    return ["cuda:0","cuda:0"]



def main(args):

    python_log.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=python_log.WARNING)

    has_test_split = True
    devices = get_devices(args.cuda)
    kg = "cpnet"

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.resume_checkpoint is not None and args.resume_checkpoint != "None"
    wandb_id = args.resume_id if resume else wandb.util.generate_id()
    args.wandb_id = wandb_id

    args.hf_version = transformers.__version__
    print('run_name: ',args.run_name)
    with wandb.init(project="KG-LM", config=args, name=args.run_name, resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode):
        print(socket.gethostname())
        print ("pid:", os.getpid())
        print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
        # print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
        utils.print_cuda_info()
        print("wandb id: ", wandb_id)

        if args.mode == 'train':
            train(args, resume, has_test_split, devices, kg)
        elif "eval" == args.mode:
            evaluate(args, has_test_split, devices, kg)
        elif "eval_zero_shot" == args.mode:
            evaluate_zero_shot(args, has_test_split, devices, kg)
        elif "eval_ddp" == args.mode:
            evaluate_ddp(args, has_test_split, devices, kg)
        elif "eval_lora" == args.mode:
            evaluate_lora(args, has_test_split, devices, kg)
        else:
            raise ValueError('Invalid mode')


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()




    # General
    parser.add_argument('--merge_lora_test',default=False,type=utils.bool_flag,help='Whether to merge the lora test')
    parser.add_argument('--lora',default=True,type=utils.bool_flag,help='Whether to use LoRA.')
    parser.add_argument('--num_context', default=5, type=int, help='The number of context nodes to consider.')
    parser.add_argument('--label_content_type', default='text_google', choices=['text_google','text', 'index', 'text_index','text_google_number'], help='The type of the labels for T5.')
    parser.add_argument('--decoder_only',default=False,type=utils.bool_flag,help='Whether to use a decoder-only model')
    parser.add_argument('--ddp',default=False,type=utils.bool_flag,help='Whether to use DistributedDataParallel.')
    parser.add_argument('--use_trainer', default=False, type=utils.bool_flag, help='Whether to use the Hugging Face Trainer.')
    parser.add_argument('--gradient_checkpointing', default=False, type=utils.bool_flag, help='Whether to use gradient checkpointing.')
    parser.add_argument('--train_header',default=False,type=utils.bool_flag,help='Whether to use the header of the llm.')
    parser.add_argument('--unfreeze_infuse', default=True, type=utils.bool_flag, help='Whether to unfreeze the infuse layer.')
    parser.add_argument('-k', '--k', default=2, type=int, help='The number of GreaseLM layers')
    parser.add_argument('--mode', default='train', choices=['train', 'eval_ddp','eval','eval_lora','eval_zero_shot'], help='run training or evaluation')
    parser.add_argument("--max_new_tokens", default=20, type=int, help="The number of new tokens to generate.")
    parser.add_argument("--do_sample",default=False,type=utils.bool_flag,help='whether to use do_sample')
    parser.add_argument("--num_beams",default=1,type=int,help='num of the beams')
    parser.add_argument("--lm_label_type", default="all_query", type=str, help="The type of the labels for the language model. only_label: only model label, or all_query: model the question and answer together.")
    parser.add_argument("--frozen_lm", default=False, type=utils.bool_flag, help="Whether to use a frozen language model.")
    
    
    parser.add_argument('--context_embedding_zero', default=True, type=utils.bool_flag, help='Whether to use zero embeddings for the context nodes.')
    parser.add_argument('--context_q_a_link_strong', default=True, type=utils.bool_flag, help='Whether to use the strong context question-answer link.')
    parser.add_argument('--edge_class', default='soft', type=str, help='The class of the edges in the KG.')
    parser.add_argument('--use_edge_score',default=True,type=utils.bool_flag,help='Whether to use the edge score to calculate the similarity between the question and the relation.')
    parser.add_argument('--question_rel_similarity',default='BiLinearSimilarity',type=str,help='The similarity function used to calculate the similarity between the question and the relation.')
    parser.add_argument("--num_key_value",default=5,type=int,help="The number of key-value pairs to infuse llm from gnn layers")
    parser.add_argument('--infuse_layer',default=[22,23],type=int,nargs='+',help='The layers to infuse the KG information.')

    parser.add_argument('--save_dir', default=f'./saved_models/greaselm/'+time_now+'/', help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag, help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default='runs/arc/greaselm__ds_arc__enc__k0__sd5__20240604_163613/model.pt', help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('--load_model_state_dict',default="output/checkpoint-620/pytorch_model.bin",type=str,help="The model state_dict path to load in the evaluation mode.")
    parser.add_argument('--load_train_args',default="output/args_ours.pt",type=str,help="The training args path to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", default="greaselm__ds_obqa__enc_obqa__k3__sd5__iedim400__unfrz0__"+time_now,type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/re_graph/train.re_graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/re_graph/dev.re_graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/re_graph/test.re_graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture

    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=200, type=int, help='dimension of the GNN layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')

    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")


    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=2, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--lora_r', default=8, type=int, help='The number of attention heads in the LoRA layer.')
    parser.add_argument('--lora_alpha',default=16,type=int,help='The alpha parameter in the LoRA layer.')
    parser.add_argument('--lora_dropout',default=0.05,type=float,help='The dropout rate in the LoRA layer.')
    args = parser.parse_args()

    if args.ddp==True:
        args.freeze_ent_emb=False
        args.use_trainer=True
    else:
        args.freeze_ent_emb=True
        args.use_trainer=False
    main(args)
