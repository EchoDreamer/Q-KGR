multi_gpu_debug=False

import itertools
import json
import pickle
import os
import transformers
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass
from collections.abc import Mapping

from torch.utils.data import Dataset
from transformers import (
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP)



try:
    from transformers import (LlamaTokenizer,T5Tokenizer)
except:
    from transformers import (T5Tokenizer)
    
    
from transformers import PreTrainedTokenizer
from preprocess_utils import conceptnet
from utils import utils


MODEL_CLASS_TO_NAME = {
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}


TRAINING_ARGS_NAME = "training_args.bin"
MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


model_name__='huggingface/model/llama2-chat-7B'
MODEL_NAME_TO_CLASS[model_name__] = 'llama'
model_name__='huggingface/model/Llama-2-7b-hf'
MODEL_NAME_TO_CLASS[model_name__] = 'llama'


MODEL_NAME_TO_CLASS["huggingface/model/flan-t5-xl"]='T5'
MODEL_NAME_TO_CLASS["huggingface/model/flan-t5-xxl"]='T5'
Llmaa_SPECIAL_TOKENS = dict()
Llmaa_SPECIAL_TOKENS['pad_token'] = '<PAD>'


def batch_dict(list_dict) -> dict:
    _batch_dict = {}
    keys = list_dict[0].keys()
    for key in keys:
        _batch_dict[key] = [_dict[key] for _dict in list_dict]
    return _batch_dict
ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]  





@dataclass
class ComputeMetrics:
    tokenizer: "PreTrainedTokenizer"

    def __call__(self,eval_preds):
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        origin_length=labels.shape[1]
        count_correct=0.0
        count_sum=0.0
        if True:
            result = preds
            token_result=self.tokenizer.batch_decode(result[:,origin_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            replaced_array = np.where(labels == -100, np.array(2), labels).reshape(result.shape[0], -1)
            token_label=self.tokenizer.batch_decode(replaced_array, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            count_sum+=labels.shape[0]

            for sub_label,sub_result in zip (token_label,token_result):
                # if log_writer is not None:
                #     log_writer.write('{},{}\n'.format(sub_label,sub_result ))
                if sub_label.lower() in sub_result.lower():
                    count_correct+=1
        return_dict={
            "acc":count_correct/count_sum
        }
        return return_dict
        
@dataclass
class DataCollatorForDreamLLMDataset:
    def __call__(self, features): # 输入是list[dict]
        # examples = batch_dict(examples)
        # return examples



        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}


        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            ##################TODO:但是这里有一个问题是device怎么确定的
            if k in ('edge_index','edge_type'):
                batch[k] = [f[k] for f in features]
            elif k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch

class MultiGPUSparseAdjDataBatchGenerator_(Dataset):
    def __init__(self,qids,labels,tensors0, tensors1, adj_data=None):
        if multi_gpu_debug:
            count=80
            self.qids = qids[:count]
            self.labels = tuple([xx[:count] for xx in labels])
            self.tensors0 = tuple([xx[:count] for xx in tensors0])
            self.tensors1 =[xx[:count] for xx in tensors1]
            self.adj_data = tuple([xx[:count] for xx in adj_data])
        else:
            self.qids = qids
            self.labels = labels
            self.tensors0 = tensors0
            self.tensors1 = tensors1
            self.adj_data = adj_data
    def __len__(self):
        return len(self.qids)
    

        
    def __getitem__(self, index):
        
        edge_index_all, edge_type_all = self.adj_data
        
        #对于tensor数据
        if isinstance(self.labels,tuple):
            single_label = [x[index] for x in self.labels]   
        else:
            single_label = self.labels[index]
        single_tensors0 = [x[index] for x in self.tensors0]
        single_tensors1 = [x[index] for x in self.tensors1]
        # concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask
        # print('__getitem__')
        return_dict={
            # 'qids':self.qids[index],            #str
            'labels':single_label[0] if isinstance(single_label,list) else single_label,         #tensor
            'input_ids':single_tensors0[0],     #tensor
            'attention_mask':single_tensors0[1],#tensor 
            # 'tensors0':single_tensors0,         #list[tensor]
            'concept_ids':single_tensors1[0],   #tensor
            'node_type_ids':single_tensors1[1], #tensor
            'node_scores':single_tensors1[2],   #tensor
            'adj_lengths':single_tensors1[3],   #tensor
            'special_nodes_mask':single_tensors1[4], #tensor
            # 'tensors1':single_tensors1,         #list[tensor]
            'edge_index':edge_index_all[index], #tensor
            'edge_type':edge_type_all[index],   #tensor
        }
        return return_dict

class MultiGPUSparseAdjDataBatchGenerator_T5(Dataset):
    def __init__(self,qids,labels,tensors0, tensors1, adj_data=None):
        if multi_gpu_debug:
            count=80
            self.qids = qids[:count]
            self.labels = tuple([xx[:count] for xx in labels])
            self.tensors0 = tuple([xx[:count] for xx in tensors0])
            self.tensors1 =[xx[:count] for xx in tensors1]
            self.adj_data = tuple([xx[:count] for xx in adj_data])
        else:
            self.qids = qids
            self.labels = labels
            self.tensors0 = tensors0
            self.tensors1 = tensors1
            self.adj_data = adj_data
    def __len__(self):
        return len(self.qids)
    

        
    def __getitem__(self, index):
        
        edge_index_all, edge_type_all = self.adj_data
        
        #对于tensor数据
        if isinstance(self.labels,tuple):
            single_label = [x[index] for x in self.labels]  
        else:
            single_label = self.labels[index]
        single_tensors0 = [x[index] for x in self.tensors0]
        single_tensors1 = [x[index] for x in self.tensors1]
        # concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask
        # print('__getitem__')
        

        return_dict={
            # 'qids':self.qids[index],            #str
            'labels':single_label[0] if isinstance(single_label,list) else single_label,         #tensor
            'encoder_input_ids':single_tensors0[0],     #tensor
            'encoder_attention_mask':single_tensors0[1],#tensor 
            'decoder_input_ids':single_tensors0[2],
            'decoder_attention_mask':single_tensors0[3],
            # 'tensors0':single_tensors0,         #list[tensor]
            'concept_ids':single_tensors1[0],   #tensor
            'node_type_ids':single_tensors1[1], #tensor
            'node_scores':single_tensors1[2],   #tensor
            'adj_lengths':single_tensors1[3],   #tensor
            'special_nodes_mask':single_tensors1[4], #tensor
            # 'tensors1':single_tensors1,         #list[tensor]
            'edge_index':edge_index_all[index], #tensor
            'edge_type':edge_type_all[index],   #tensor
        }
        return return_dict

class MultiGPUSparseAdjDataBatchGenerator(object):         
    """A data generator that batches the data and moves them to the corresponding devices."""
    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0   
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1
    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            # batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_labels = [self._to_device(x[batch_indexes], self.device1) for x in self.labels]
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_tensors1[0] = batch_tensors1[0].to(self.device0)
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_labels[0], *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])
            #list,list

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class GreaseLM_DataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, n_train=-1, debug=False, cxt_node_connects_all=False, kg="cpnet",
                 lm_label_type='all_query',dataset_name='obqa',ddp=False,label_content_type='text',num_context=1):
        super().__init__()
        self.label_content_type=label_content_type
        self.ddp=ddp
        self.dataset_name=dataset_name
        if dataset_name=='obqa':
            self.num_choice=4
        elif dataset_name=='riddle':
            self.num_choice=5
        elif dataset_name=='arc':
            self.num_choice=4
        elif dataset_name=='piqa':
            self.num_choice=2
        else:
            raise ValueError('must assign self.num_choice for new datset')
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.debug = debug
        self.model_name = model_name
        self.max_node_num = max_node_num
        self.debug_sample_size = 32
        self.cxt_node_connects_all = cxt_node_connects_all

        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.load_resources(kg)

        print ('train_statement_path', train_statement_path)
        
        self.train_qids, self.train_labels, self.train_encoder_data = self.load_input_tensors(train_statement_path, max_seq_length,lm_label_type,test_data=False,label_content_type=self.label_content_type)

        *self.train_decoder_data, self.train_adj_data = self.load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num,num_context=num_context)        

        self.dev_qids, self.dev_labels, self.dev_encoder_data = self.load_input_tensors(dev_statement_path, max_seq_length,lm_label_type,test_data=True,label_content_type=self.label_content_type)

        *self.dev_decoder_data, self.dev_adj_data = self.load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,num_context=num_context)
        
        
        
        # if not debug:
        #     assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        self.dev_train_qids, self.dev_train_labels, self.dev_train_encoder_data = self.load_input_tensors(dev_statement_path, max_seq_length,lm_label_type,test_data=False,label_content_type=self.label_content_type)




        print("Finish loading dev data.")

        # Load test data
        if test_statement_path is not None:
            self.test_qids, self.test_labels, self.test_encoder_data = self.load_input_tensors(test_statement_path, max_seq_length,lm_label_type,test_data=True,label_content_type=label_content_type)

            *self.test_decoder_data, self.test_adj_data = self.load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num,num_context=num_context)
            
            print("Finish loading test data.")

        assert 0. < subsample <= 1.
        if subsample < 1. or n_train >= 0:
            # n_train will override subsample if the former is not None
            if n_train == -1:
                n_train = int(self.train_size() * subsample)
            assert n_train > 0

            if True:
                self.train_qids = self.train_qids[:n_train]
                # self.train_labels = self.train_labels[:n_train]
                self.train_labels = [x[:n_train] for x in self.train_labels]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                # assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def train_dataset(self):
        if self.model_type=='llama':
            return MultiGPUSparseAdjDataBatchGenerator_(self.train_qids, self.train_labels, tensors0=self.train_encoder_data, 
                                                    tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        elif self.model_type=='T5':
            return MultiGPUSparseAdjDataBatchGenerator_T5(self.train_qids, self.train_labels, tensors0=self.train_encoder_data, 
                                                    tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            raise ValueError('model_type must be llama or T5')

    def train(self):
        if self.debug:
            train_indexes = torch.arange(self.debug_sample_size)
        # elif self.is_inhouse:
            
        #     n_train = self.inhouse_train_indexes.size(0)
        #     train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, 
                                                   self.train_qids, self.train_labels, tensors0=self.train_encoder_data, 
                                                   tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev_dataset(self):
        if self.model_type=='llama':
            return MultiGPUSparseAdjDataBatchGenerator_(self.dev_train_qids, self.dev_train_labels, tensors0=self.dev_train_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)
        elif self.model_type=='T5':
            return MultiGPUSparseAdjDataBatchGenerator_T5(self.dev_train_qids, self.dev_train_labels, tensors0=self.dev_train_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)
        else:
            raise ValueError('model_type must be llama or T5')
    def dev(self):
        if self.debug:
            dev_indexes = torch.arange(self.debug_sample_size)
        else:
            dev_indexes = torch.arange(len(self.dev_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, dev_indexes, self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)
    def test_dataset(self):
        if self.model_type=='llama':
            return MultiGPUSparseAdjDataBatchGenerator_(self.test_qids, self.test_labels,tensors0=self.test_encoder_data, 
                                                    tensors1=self.test_decoder_data, adj_data=self.test_adj_data)
        elif self.model_type=='T5':
            return MultiGPUSparseAdjDataBatchGenerator_T5(self.test_qids, self.test_labels,tensors0=self.test_encoder_data, 
                                                    tensors1=self.test_decoder_data, adj_data=self.test_adj_data)   
        else:
            raise ValueError('model_type must be llama or T5')

        
    def test(self):
        if self.debug:
            test_indexes = torch.arange(self.debug_sample_size)

        else:
            test_indexes = torch.arange(len(self.test_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, test_indexes, self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)

    def load_resources(self, kg):


        try:
            tokenizer_class = { 'llama': LlamaTokenizer,'T5':T5Tokenizer}.get(self.model_type)
        except:
            tokenizer_class = { 'T5':T5Tokenizer}.get(self.model_type)
        
                
    
        tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer

        if kg == "cpnet":
            # Load cpnet
            cpnet_vocab_path = "data/cpnet/concept.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = conceptnet.merged_relations
        elif kg == "ddb":
            cpnet_vocab_path = "data/ddb/vocab.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = [
                'belongstothecategoryof',
                'isacategory',
                'maycause',
                'isasubtypeof',
                'isariskfactorof',
                'isassociatedwith',
                'maycontraindicate',
                'interactswith',
                'belongstothedrugfamilyof',
                'child-parent',
                'isavectorfor',
                'mabeallelicwith',
                'seealso',
                'isaningradientof',
                'mabeindicatedby'
            ]
        else:
            raise ValueError("Invalid value for kg.")

    def load_input_tensors(self, input_jsonl_path, max_seq_length,lm_label_type='all_query',test_data=False,label_content_type='text_index'):
        """Construct input tensors for the LM component of the model."""

        if self.model_type in ('T5',):
            if label_content_type=='text':
                if test_data:
                    cache_path = input_jsonl_path + "-for-test-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.loaded_cache'
                else:
                    cache_path = input_jsonl_path + "-for-train-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.loaded_cache'
            else:
                if test_data:
                    cache_path = input_jsonl_path + "-for-test-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.{label_content_type}.loaded_cache'
                else:
                    cache_path = input_jsonl_path + "-for-train-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.{label_content_type}.loaded_cache'

                
        else:
            if test_data:
                cache_path = input_jsonl_path + "-for-test-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.loaded_cache'
            else:
                cache_path = input_jsonl_path + "-for-train-sl{}".format(max_seq_length) + '.'+self.model_type +'single'+f'.{lm_label_type}.loaded_cache'

        use_cache = True

        if not self.ddp and not os.path.exists(cache_path):
            use_cache = False

        if use_cache:
            with open(cache_path, 'rb') as f:
                input_tensors = utils.CPU_Unpickler(f).load()
        else:
            if self.model_type in ('llama',):
                input_tensors = load_llama_input_tensors(input_jsonl_path, max_seq_length, self.tokenizer,lm_label_type, test_data,dataset_name=self.dataset_name,model_name=self.model_name)
            elif self.model_type in ('T5',):
                input_tensors = load_t5_input_tensors(input_jsonl_path, max_seq_length, self.tokenizer,lm_label_type, test_data,dataset_name=self.dataset_name,model_name=self.model_name,label_content_type=label_content_type)
            utils.save_pickle(input_tensors, cache_path)
            
                
        return input_tensors

    def load_sparse_adj_data_with_contextnode(self, adj_pk_path, max_node_num, num_context=1,concepts_by_sents_list=None):
        """Construct input tensors for the GNN component of the model."""
        print("Loading sparse adj data...")

        cache_path = adj_pk_path + "-nodenum{}".format(max_node_num) + ("-cntsall" if self.cxt_node_connects_all else "") +'single'+f'.num_context{num_context}.loaded_cache'

        use_cache = True

        if use_cache and not os.path.exists(cache_path):
            use_cache = False
        num_relation=0
        if use_cache:
            with open(cache_path, 'rb') as f:
                adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask = utils.CPU_Unpickler(f).load()
        else:
            # Set special nodes and links
            context_node = 0

            n_special_nodes = 1

            cxt2qlinked_rel = 0
            cxt2alinked_rel = 1

            half_n_rel = len(self.id2relation) + 2

            if self.cxt_node_connects_all:
                cxt2other_rel = half_n_rel
                half_n_rel += 1

            adj_concept_pairs = []
            with open(adj_pk_path, "rb") as in_file:
                try:
                    while True:
                        ex = pickle.load(in_file)
                        if type(ex) == dict:
                            adj_concept_pairs.append(ex)
                        elif type(ex) == list:
                            adj_concept_pairs.extend(ex)
                        else:
                            raise TypeError("Invalid type for ex.")
                except EOFError:
                    pass

            n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
            edge_index, edge_type = [], []
            adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
            concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
            node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
            node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
            special_nodes_mask = torch.zeros(n_samples, max_node_num, dtype=torch.bool)

            adj_lengths_ori = adj_lengths.clone()
            if not concepts_by_sents_list:
                concepts_by_sents_list = itertools.repeat(None)
            for idx, (_data, cpts_by_sents) in tqdm(enumerate(zip(adj_concept_pairs, concepts_by_sents_list)), total=n_samples, desc='loading adj matrices'):
                if self.debug and idx >= self.debug_sample_size:
                    break
                adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
                #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
                #concepts: np.array(num_nodes, ), where entry is concept id 
                #qm: np.array(num_nodes, ), where entry is True/False
                #am: np.array(num_nodes, ), where entry is True/False
                assert len(concepts) == len(set(concepts))
                qam = qm | am
                #sanity check: should be T,..,T,F,F,..F
                assert qam[0] == True
                F_start = False
                for TF in qam:
                    if TF == False:
                        F_start = True
                    else:
                        assert F_start == False

                assert n_special_nodes <= max_node_num
                #仅仅mask第一个特殊的token
                special_nodes_mask[idx, :n_special_nodes] = 1
                num_concept = min(len(concepts) + n_special_nodes, max_node_num) #this is the final number of nodes including contextnode but excluding PAD
                adj_lengths_ori[idx] = len(concepts)
                adj_lengths[idx] = num_concept

                #Prepare nodes
                concepts = concepts[:num_concept - n_special_nodes]
                #####TODO:这里我们依旧保留这个加一，后面去根据节点id找embedding的时候也能够对上，但是那里可能要改一下，比如看看是否需要减一
                concept_ids[idx, n_special_nodes:num_concept] = torch.tensor(concepts + 1)  #To accomodate contextnode, original concept_ids incremented by 1

                concept_ids[idx, 0] = context_node #this is the "concept_id" for contextnode

                #Prepare node scores
                if cid2score is not None:
                    if -1 not in cid2score:
                        cid2score[-1] = 0
                    for _j_ in range(num_concept):
                        _cid = int(concept_ids[idx, _j_]) - 1 # Now context node is -1
                        node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

                #Prepare node types

                node_type_ids[idx, 0] = 3 # context node
                node_type_ids[idx, 1:n_special_nodes] = 4 # sent nodes
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept - n_special_nodes]] = 0
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept - n_special_nodes]] = 1

                #Load adj
                ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
                k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
                n_node = adj.shape[1]
                assert len(self.id2relation) == adj.shape[0] // n_node
                i, j = ij // n_node, ij % n_node
                extra_i, extra_j, extra_k = [], [], []
                #Prepare edges
                if True:
                    i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                    for _coord, q_tf in enumerate(qm):
                        _new_coord = _coord + n_special_nodes
                        if _new_coord > num_concept:
                            break
                        if q_tf:
                            extra_i.append(cxt2qlinked_rel) #rel from contextnode to question concept
                            extra_j.append(0) #contextnode coordinate
                            extra_k.append(_new_coord) #question concept coordinate
                        elif self.cxt_node_connects_all:
                            extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                            extra_j.append(0) #contextnode coordinate
                            extra_k.append(_new_coord) #other concept coordinate
                    for _coord, a_tf in enumerate(am):
                        _new_coord = _coord + n_special_nodes
                        if _new_coord > num_concept:
                            break
                        if a_tf:
                            extra_i.append(cxt2alinked_rel) #rel from contextnode to answer concept
                            extra_j.append(0) #contextnode coordinate
                            extra_k.append(_new_coord) #answer concept coordinate
                        elif self.cxt_node_connects_all:
                            extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                            extra_j.append(0) #contextnode coordinate
                            extra_k.append(_new_coord) #other concept coordinate

                # half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)


                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                #计算边的数量e
                # if i.size(0)>num_relation:
                #     num_relation=i.size(0)
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]

            
            # print(num_relation)
            if not self.debug:
                with open(cache_path, 'wb') as f:
                    pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask], f)
            

        ori_adj_mean  = adj_lengths_ori.float().mean().item()
        ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
        print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
            ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
            ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                        (node_type_ids == 1).float().sum(1).mean().item()))
        #这个操作还是要好好看看
        # edge_index = list(map(list, zip(*(iter(edge_index),) ))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
        # edge_type = list(map(list, zip(*(iter(edge_type),)))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

        concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [x.view(-1, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)]
        #concept_ids: (n_questions, num_choice, max_node_num)
        #node_type_ids: (n_questions, num_choice, max_node_num)
        #node_scores: (n_questions, num_choice, max_node_num)
        #adj_lengths: (n_questions,　num_choice)
        return concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, (edge_index, edge_type) #, half_n_rel * 2 + 1
    
def load_t5_input_tensors(statement_jsonl_path, max_seq_length, tokenizer:transformers.PreTrainedTokenizer,lm_label_type='all_query',test_data=False,dataset_name='obqa',model_name=None,label_content_type='index'):
    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)
    def pre_process_datasets(encoded_datasets,label_length, max_seq_length, pad_tokens_ids,encoded_label):
        ###########TODO:需要注意一下train的时候encoder输入部分是否也要有1结尾
        if True:
            tensor_datasets = []
            for dataset,sub_label_length,sub_encoded_label in zip(encoded_datasets,label_length,encoded_label):
                n_batch = len(dataset)
                encoder_input_ids=np.full((n_batch, max_seq_length), fill_value=pad_tokens_ids, dtype=np.int64)
                encoder_input_mask = np.zeros((n_batch, max_seq_length), dtype=np.int64)
                decoder_input_ids=np.full((n_batch, max_seq_length), fill_value=pad_tokens_ids, dtype=np.int64)
                decoder_input_mask=np.zeros((n_batch, max_seq_length), dtype=np.int64)
                lm_labels = np.full((n_batch, max_seq_length), fill_value=-100, dtype=np.int64)
                labels_position=np.zeros((n_batch,2),dtype=np.int64)    #前闭后开
                for i,data in enumerate(dataset):
                    q = data
                    _truncate_seq_pair(q,"", max_seq_length )
                    aaaa=sub_encoded_label[i]
                    _truncate_seq_pair(aaaa,"", max_seq_length )
                    qa = q
                    encoder_input_ids[i, :len(qa)] = qa
                    encoder_input_mask[i, :len(qa)] = 1
                    labels_position[i][0]=0
                    labels_position[i][1]=sub_label_length[i]-1
                    #decoder输入开头是0，结尾需要把</s>去掉
                    decoder_input_ids[i,:sub_label_length[i]]=[0]+aaaa[:-1]
                    decoder_input_mask[i,:sub_label_length[i]]=1
                    lm_labels[i,:sub_label_length[i]]=aaaa  
                all_inputs = (encoder_input_ids, encoder_input_mask, decoder_input_ids,decoder_input_mask,lm_labels,labels_position)
                tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
            return tensor_datasets
       
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    def prompt_generate(dataset,test_data=False,label_content_type='index'):
        file_name = model_name.split('/')[-1]
        if file_name=='flan-t5-xl' or file_name=='flan-t5-xxl':
            if dataset_name=='piqa':
                if label_content_type=='text_google': 
                    questions_new=[data__[0] for data__ in dataset]
                    def format_options(options):
                        return 'OPTIONS:\n- ' + '\n- '.join(options)
                    options_= [format_options(data_[1:]) for data_ in dataset]
                    prompt_list=[]
                    for i in range (len(dataset)):
                        prompt=f"Here is a goal: {questions_new[i]}\n\nWhich way makes more sense to accomplish this goal?\n\n{options_[i]}\nAnswer: "
                        prompt_list.append(prompt)
                    return prompt_list
                elif label_content_type=='text_google_number':
                    questions_new=[data__[0] for data__ in dataset]
                    def format_options(options):
                        options=[f"{i}:{options[i]}" for i in range(len(options))]
                        return 'OPTIONS:\n- ' + '\n- '.join(options)
                    options_= [format_options(data_[1:]) for data_ in dataset]
                    prompt_list=[]
                    for i in range (len(dataset)):
                        prompt=f"Here is a goal: {questions_new[i]}\n\nWhich way makes more sense to accomplish this goal?\n\n{options_[i]}\nAnswer: "
                        prompt_list.append(prompt)
                    return prompt_list
            else:
                if label_content_type=='text_google': 
                    questions_new=[data__[0] for data__ in dataset]
                    def format_options(options):
                        return 'OPTIONS:\n- ' + '\n- '.join(options)
                    options_= [format_options(data_[1:]) for data_ in dataset]
                    prompt_list=[]
                    for i in range (len(dataset)):
                        prompt=f"{questions_new[i]}\n\n{options_[i]}\nAnswer: "
                        prompt_list.append(prompt)
                    return prompt_list
                elif label_content_type=='text_google_number':
                    questions_new=[data__[0] for data__ in dataset]
                    def format_options(options):
                        options=[f"{i}:{options[i]}" for i in range(len(options))]
                        return 'OPTIONS:\n- ' + '\n- '.join(options)
                    options_= [format_options(data_[1:]) for data_ in dataset]
                    prompt_list=[]
                    for i in range (len(dataset)):
                        prompt=f"{questions_new[i]}\n\n{options_[i]}\nAnswer: "
                        prompt_list.append(prompt)
                    return prompt_list
    padding_position='right'
    tokenizer.padding_side =padding_position                  
    #我们需要分别给encoder和decoder准备数据，从generate生成情况看，decoder部分需要前面加上0，而不是加1；encoder部分前面不加东西

    padding_ids=0
    dataset = load_qa_dataset(statement_jsonl_path)   #[tuple], where each tuple is (id, question, choice1, choice2, choice3, choice4, choice5, label{int})
    
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2
    if label_content_type=='text_google':
        label=[data[data[-1]+1] for data in dataset]
    elif label_content_type=='text_google_number':
        label=[f"{data[-1]}:{data[data[-1]+1]}" for data in dataset]
    else:
        raise ValueError("Invalid value for label_content_type")
    dataset = [data[:-1] for data in dataset] # discard labels
    #这里面的数据用于encoder数据编码
    dataset=prompt_generate(dataset=dataset,test_data=test_data,label_content_type=label_content_type)
    encoded_dataset=[tokenizer(data__)['input_ids'] for data__ in dataset]
    #我们需要先把label给他加上0，然后再编码
    encoded_label=[tokenizer(data__)['input_ids'] for data__ in label]
    label_length=[len(sub_label) for sub_label in encoded_label]
    (encoder_input_ids, encoder_input_mask, decoder_input_ids,decoder_input_mask,lm_labels,labels_position), = pre_process_datasets([encoded_dataset],[label_length], max_seq_length, padding_ids,
                                                                                encoded_label=[encoded_label])   
    return examples_ids, (lm_labels,labels_position), (encoder_input_ids, encoder_input_mask, decoder_input_ids,decoder_input_mask)
      
def load_llama_input_tensors(statement_jsonl_path, max_seq_length, tokenizer:transformers.PreTrainedTokenizer,lm_label_type='all_query',test_data=False,dataset_name='obqa',model_name=None):
    # def load_qa_dataset(dataset_path):
    #     """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    #     with open(dataset_path, "r", encoding="utf-8") as fin:
    #         output = []
    #         for line in fin:
    #             input_json = json.loads(line)
    #             label = ord(input_json.get("answerKey", "A")) - ord("A")
    #             output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
    #     return output
    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                answer_key =input_json.get("answerKey", "A")
                mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D','5':'E'}
                if answer_key in mapping:
                    answer_key = mapping[answer_key]
                label = ord(answer_key) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]],label))
        return output
    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)
    def pre_process_datasets(encoded_datasets,label_length, num_choices, max_seq_length, pad_tokens_ids,lm_label_type,padding_position,encoded_label,test_data):
        if not test_data:
            tensor_datasets = []
            for dataset,sub_label_length in zip(encoded_datasets,label_length):
                n_batch = len(dataset)
                input_ids=np.full((n_batch, max_seq_length), fill_value=pad_tokens_ids, dtype=np.int64)
                input_mask = np.zeros((n_batch, max_seq_length), dtype=np.int64)
                lm_labels = np.full((n_batch, max_seq_length), fill_value=-100, dtype=np.int64)
                labels_position=np.zeros((n_batch,2),dtype=np.int64)    
                for i,data in enumerate(dataset):
                    q = data
                    if True:
                        _truncate_seq_pair(q,"", max_seq_length )
                        qa = q
                        if padding_position=='left':
                            input_ids[i,max_seq_length-len(qa):]=qa
                            input_mask[i,max_seq_length-len(qa):] = 1
                            labels_position[i][0]=max_seq_length-sub_label_length[i]         
                            labels_position[i][1]=max_seq_length-1
                            if lm_label_type=='all_query':
                                lm_labels[i, max_seq_length-len(qa):max_seq_length-1] = qa[1:]
                            elif lm_label_type=='only_label':
                                lm_labels[i,max_seq_length-2-sub_label_length[i]:max_seq_length-1]=qa[len(qa)-sub_label_length[i]-1:]

                            else:
                                raise ValueError("Invalid value for lm_label_type")
                        elif padding_position=='right':
                            input_ids[i, :len(qa)] = qa
                            input_mask[i, :len(qa)] = 1
                            labels_position[i][0]=len(qa)-1-sub_label_length[i]
                            labels_position[i][1]=len(qa)-1
                            if lm_label_type=='all_query':
                                lm_labels[i, :len(qa)-1] = qa[1:]
                            elif lm_label_type=='only_label':
                                lm_labels[i,len(qa)-2-sub_label_length[i]:len(qa)-1]=qa[len(qa)-sub_label_length[i]-1:]
                            else:
                                raise ValueError("Invalid value for lm_label_type")
                all_inputs = (input_ids, input_mask, lm_labels,labels_position)
                tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
            return tensor_datasets
        else:
            tensor_datasets = []
            for dataset,sub_label_length,sub_label in zip(encoded_datasets,label_length,encoded_label):
                n_batch = len(dataset)
                input_ids=np.full((n_batch, max_seq_length), fill_value=pad_tokens_ids, dtype=np.int64)
                input_mask = np.zeros((n_batch, max_seq_length), dtype=np.int64)
                lm_labels = np.full((n_batch, 100), fill_value=pad_tokens_ids, dtype=np.int64)
                labels_position=np.zeros((n_batch,2),dtype=np.int64)    
                for i,data in enumerate(dataset):
                    q = data
                    if True:
                        _truncate_seq_pair(q,"", max_seq_length )
                        qa = q
                        if padding_position=='left':
                            input_ids[i,max_seq_length-len(qa):]=qa
                            input_mask[i,max_seq_length-len(qa):] = 1
                            # labels_position[i][j][0]=max_seq_length-1-sub_label_length[i]
                            # labels_position[i][j][1]=max_seq_length-1
                            lm_labels[i,:len(sub_label[i])]=sub_label[i]

                        elif padding_position=='right':
                            input_ids[i, :len(qa)] = qa
                            input_mask[i, :len(qa)] = 1

                            lm_labels[i,:len[sub_label[i]]]=sub_label[i]

                all_inputs = (input_ids, input_mask, lm_labels,labels_position)
                tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
            return tensor_datasets
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    def prompt_generate(dataset,label,num_choices,test_data=False):
        prompt_header="<s> Please answer the following question based on the given answers.\n"
        flag_prompt_header=True
        flag_key=True
        file_name = model_name.split('/')[-1]

        if file_name=='Llama-2-7b-hf':

            if dataset_name=='obqa':  
                flag_prompt_header=True
                flag_key=False
            else:
                raise ValueError("Invalid value for dataset_name")
        elif file_name=='llama2-chat-7B':

            if dataset_name=='obqa':
                flag_prompt_header=False
                flag_key=True
            else:
                raise ValueError("Invalid value for dataset_name")
        
        prompt_list=[]
        if not test_data:
            for i in range (len(dataset)):
                letter_number_dict={0:'A',1:'B',2:'C',3:'D',4:'E'}
                candidate_answer=dataset[i][1:]
                letter_label=[letter_number_dict[ii] for ii in range (num_choices-1)]
                if flag_key:
                    candidate_answer=[letter_label[j]+': '+candidate_answer[j] for j in range(num_choices-1)]

                answers='", "'.join(candidate_answer)
                if flag_key:
                    answers='"'+answers+f'", or "{letter_number_dict[num_choices-1]}: '+dataset[i][-1]+'"'
                else:
                    answers='"'+answers+'", or "'+dataset[i][-1]+'"'
                

                candidate_prefix="Answer with only "+answers+"."
                if flag_prompt_header:
                    prompt = f"{prompt_header}Question: {dataset[i][0]}\n"+candidate_prefix+f" The best answer is {label[i]} </s>"  
                else:
                    prompt = f"<s> Question: {dataset[i][0]}\n"+candidate_prefix+f" The best answer is {label[i]} </s>"
                # print(prompt)
                prompt_list.append(prompt)
            return prompt_list
        else:
            for i in range (len(dataset)):
                letter_number_dict={0:'A',1:'B',2:'C',3:'D',4:'E'}
                candidate_answer=dataset[i][1:]
                letter_label=[letter_number_dict[ii] for ii in range (num_choices-1)]
                if flag_key:
                    candidate_answer=[letter_label[j]+': '+candidate_answer[j] for j in range(num_choices-1)]

                answers='", "'.join(candidate_answer)
                if flag_key:
                    answers='"'+answers+f'", or "{letter_number_dict[num_choices-1]}: '+dataset[i][-1]+'"'
                else:
                    answers='"'+answers+'", or "'+dataset[i][-1]+'"'

                candidate_prefix="Answer with only "+answers+"."
                if flag_prompt_header:
                    prompt = f"{prompt_header}Question: {dataset[i][0]}\n"+candidate_prefix+f" The best answer is"  
                else:
                    prompt = f"<s> Question: {dataset[i][0]}\n"+candidate_prefix+f" The best answer is"
                # print(prompt)
                prompt_list.append(prompt)
            return prompt_list

    padding_position='left'
    tokenizer.padding_side =padding_position                  
    
    eos_token=tokenizer.eos_token
    padding_ids=tokenizer.convert_tokens_to_ids(eos_token)
    dataset = load_qa_dataset(statement_jsonl_path)   #[tuple], where each tuple is (id, question, choice1, choice2, choice3, choice4, choice5, label{int})
    
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2
    label=[data[data[-1]+1] for data in dataset]
    dataset = [data[:-1] for data in dataset] # discard labels

    dataset=prompt_generate(dataset=dataset,label=label,num_choices=num_choices,test_data=test_data)
    encoded_dataset = tokenize_and_encode(tokenizer,dataset)
    encoded_label=tokenize_and_encode(tokenizer,label)
    label_length=[len(sub_label) for sub_label in encoded_label]
    (input_ids, input_mask, lm_labels,labels_position), = pre_process_datasets([encoded_dataset],[label_length], num_choices, max_seq_length, padding_ids,lm_label_type,
                                                                               padding_position=padding_position,encoded_label=[encoded_label],test_data=test_data)
    return examples_ids, (lm_labels,labels_position), (input_ids, input_mask)