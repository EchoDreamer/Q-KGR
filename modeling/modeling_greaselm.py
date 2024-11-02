import logging as python_log

import torch.nn.init as init
from peft import get_peft_model
import torch

import torch.nn as nn
import torch.nn.functional as F
# from transformers import PretrainedConfig
from transformers import LlamaForCausalLM,T5ForConditionalGeneration
# from transformers.utils.hub import is_remote_url
from modeling import modeling_gnn
# from modeling.file_gnn_vector import 
from utils import layers
from utils.utils import freeze_net,unfreeze_net
from utils import utils
from utils.data_utils import MODEL_NAME_TO_CLASS
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
# from transformers.models.llama.modeling_llama import StaticCache,DynamicCache
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from torch.utils.checkpoint import checkpoint

logger = python_log.getLogger(__name__)

ModelClass=LlamaForCausalLM
import warnings
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
from transformers.modeling_outputs import BaseModelOutput,BaseModelOutputWithPastAndCrossAttentions,Seq2SeqLMOutput
# print ('ModelClass', ModelClass)
class GreaseLM(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, infuse_layer=-1,
                 num_key_value=1,question_rel_similarity='BiLinearSimilarity',
                 use_edge_score=False,edge_class='hard',context_q_a_link_strong=True,context_embedding_zero=True,unfreeze_infuse=False,
                 train_header=False,gradient_checkpointing=False,max_new_tokens=10,lora_config=None):
        self.max_new_tokens=max_new_tokens
        super().__init__()
        self.lmgnn = LMGNN(args, model_name, k, n_ntype, n_etype,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        p_emb, p_gnn, p_fc, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range,
                                        infuse_layer=infuse_layer,num_key_value=num_key_value,
                                        question_rel_similarity=question_rel_similarity,
                                        use_edge_score=use_edge_score,edge_class=edge_class,context_q_a_link_strong=context_q_a_link_strong,
                                        context_embedding_zero=context_embedding_zero,unfreeze_infuse=unfreeze_infuse,train_header=train_header,gradient_checkpointing=gradient_checkpointing,
                                        lora_config=lora_config)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        """
        n_examples = len(edge_index_init)
        subgraph_indice=[ edge_index_init[_j_].size(-1) for _j_ in range(n_examples) ]
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        
        assert edge_index.size(-1)==sum(subgraph_indice)
        
        return edge_index, edge_type,subgraph_indice
    
    
    
    
   
    def forward(self,labels,input_ids,attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type):
        eval_acc=False


        if self.training or not eval_acc:
            lm_inputs=input_ids,attention_mask
            node_scores = torch.zeros_like(node_scores)

            edge_index, edge_type,subgraph_indice = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
            adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

            logits = self.lmgnn(lm_inputs, concept_ids,
                                        node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,
                                        emb_data=None,subgraph_indice=subgraph_indice)

            loss_func = nn.CrossEntropyLoss(reduction='mean')
            size_vocabulary=logits.size(-1)
            loss = loss_func(logits.view(-1,size_vocabulary),labels.view(-1))
            return_dict={
                'logits':logits,
                'loss':loss
            }
            return return_dict

        else:
            logits=self.generate(labels,input_ids,attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type,max_new_tokens=self.max_new_tokens,do_sample=False,num_beams=1)
            loss=torch.tensor(0.1).to(logits.device)
            return_dict={
                'logits':logits,    
                'loss':loss
            }
            return return_dict

    def generate(self, labels,input_ids,attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type,**kwargs):

        lm_inputs=input_ids,attention_mask
        node_scores = torch.zeros_like(node_scores)

        edge_index, edge_type,subgraph_indice = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]


        result=self.lmgnn.generate(lm_inputs, concept_ids,
                                    node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,
                                    emb_data=None, subgraph_indice=subgraph_indice,**kwargs)

        return result


class GreaseLM_(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, infuse_layer=-1,
                 num_key_value=1,question_rel_similarity='BiLinearSimilarity',
                 use_edge_score=False,edge_class='hard',context_q_a_link_strong=True,context_embedding_zero=True,unfreeze_infuse=False,
                 train_header=False,gradient_checkpointing=False,max_new_tokens=10,lora_config=None,lora_test_dir=None,
                 merge_lora_test=False):
        self.max_new_tokens=max_new_tokens
        super().__init__()
        self.lmgnn = LMGNN(args, model_name, k, n_ntype, n_etype,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        p_emb, p_gnn, p_fc, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range,
                                        infuse_layer=infuse_layer,num_key_value=num_key_value,
                                        question_rel_similarity=question_rel_similarity,
                                        use_edge_score=use_edge_score,edge_class=edge_class,context_q_a_link_strong=context_q_a_link_strong,
                                        context_embedding_zero=context_embedding_zero,unfreeze_infuse=unfreeze_infuse,train_header=train_header,
                                        gradient_checkpointing=gradient_checkpointing,lora_config=lora_config,lora_test_dir=lora_test_dir,
                                        merge_lora_test=merge_lora_test
                                        )

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        """
        n_examples = len(edge_index_init)
        subgraph_indice=[ edge_index_init[_j_].size(-1) for _j_ in range(n_examples) ]
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        
        assert edge_index.size(-1)==sum(subgraph_indice)
        
        return edge_index, edge_type,subgraph_indice

    def forward(self,labels,encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type):
        eval_acc=False
        """ 
        return_dict={
            # 'qids':self.qids[index],            #str
            'labels':single_label[0] if isinstance(single_label,list) else single_label,         #tensor
            'tensors0':single_tensors0,         #list[tensor]
            'tensors1':single_tensors1,         #list[tensor]
            'edge_index':edge_index_all[index], #tensor
            'edge_type':edge_type_all[index],   #tensor
        }
        """
        

        if self.training or not eval_acc:
            

            
            lm_inputs=encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask
            node_scores = torch.zeros_like(node_scores)
            edge_index, edge_type,subgraph_indice = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
            adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

            logits = self.lmgnn(lm_inputs, concept_ids,
                                        node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,
                                        emb_data=None,subgraph_indice=subgraph_indice)

            loss_func = nn.CrossEntropyLoss(reduction='mean')
            size_vocabulary=logits.size(-1)
            loss = loss_func(logits.view(-1,size_vocabulary),labels.view(-1))
            return_dict={
                'logits':logits,
                'loss':loss
            }
            return return_dict

        else:

            logits=self.generate(labels,encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type,max_new_tokens=self.max_new_tokens,do_sample=False,num_beams=1)
            loss=torch.tensor(0.1).to(logits.device)
            return_dict={
                'logits':logits,     
                'loss':loss
            }
            return return_dict

    def generate(self, labels,encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask,edge_index, edge_type,**kwargs):

        lm_inputs=encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask
        node_scores = torch.zeros_like(node_scores)
        edge_index, edge_type,subgraph_indice = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]


        result=self.lmgnn.generate(lm_inputs, concept_ids,
                                    node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,
                                    emb_data=None, subgraph_indice=subgraph_indice,**kwargs)

        return result


class LMGNN(nn.Module):
    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02,  infuse_layer=-1,
                 num_key_value=1,question_rel_similarity='BiLinearSimilarity',
                 use_edge_score=False,edge_class='hard',context_q_a_link_strong=True,context_embedding_zero=True,unfreeze_infuse=False,
                 train_header=False,gradient_checkpointing=False,lora_config=None,lora_test_dir=None,
                merge_lora_test=False):
        super().__init__()

        self.model_type=MODEL_NAME_TO_CLASS[model_name]

        self.context_embedding_zero=context_embedding_zero
        
        # self.edge_class=edge_class
        # self.question_rel_similarity=question_rel_similarity
        self.use_edge_score=use_edge_score
        self.context_q_a_link_strong=context_q_a_link_strong
        if self.context_embedding_zero and self.use_edge_score:
            self.similarity_compute_=layers.similarity_compute(question_rel_similarity,edge_class,concept_dim)

        self.init_range = init_range
        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        self.activation = layers.GELU()

        if k >0 and self.context_embedding_zero:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)


        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)
        if self.model_type=='llama':
            self.mp, self.loading_info = TextKGMessagePassing.from_pretrained(model_name, output_hidden_states=False, output_loading_info=True, args=args, 
                                                                            k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, concept_dim=concept_dim, p_fc=p_fc, 
                                                                             infuse_layer=infuse_layer,
                                                                            num_key_value=num_key_value,question_rel_similarity=question_rel_similarity,
                    use_edge_score=use_edge_score,edge_class=edge_class,context_q_a_link_strong=context_q_a_link_strong,context_embedding_zero=context_embedding_zero,n_concept=n_concept,
                    concept_in_dim=concept_in_dim,pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,p_emb=p_emb,frozen_lm=args.frozen_lm,model_type=self.model_type,
                    unfreeze_infuse=unfreeze_infuse,train_header=train_header,gradient_checkpointing=gradient_checkpointing)
        elif self.model_type=='T5':
            self.mp, self.loading_info = TextKGMessagePassing_T5.from_pretrained(model_name, output_hidden_states=False, output_loading_info=True, args=args, 
                                                                            k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, concept_dim=concept_dim, p_fc=p_fc, 
                                                                             infuse_layer=infuse_layer,
                                                                            num_key_value=num_key_value,question_rel_similarity=question_rel_similarity,
                    use_edge_score=use_edge_score,edge_class=edge_class,context_q_a_link_strong=context_q_a_link_strong,context_embedding_zero=context_embedding_zero,n_concept=n_concept,
                    concept_in_dim=concept_in_dim,pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,p_emb=p_emb,frozen_lm=args.frozen_lm,model_type=self.model_type,
                    unfreeze_infuse=unfreeze_infuse,train_header=train_header,gradient_checkpointing=gradient_checkpointing)
        else:
            raise ValueError('model_type is not supported')
        
        if lora_config is not None:
            self.mp=get_peft_model(self.mp,peft_config=lora_config)

            self.unfrozen_lora(self.mp)
        if lora_test_dir is not None:
            print('loading lora model')
            from peft import PeftModel
            self.mp=PeftModel.from_pretrained(
                self.mp,
                lora_test_dir,
                torch_dtype=torch.float32,
            )
        if merge_lora_test:   
            print('###################merge_lora_weight#################')
            merge_lora=self.mp.merge_and_unload()

            ffn1=[]
            ffn2=[]
            ffn3=[]
            for i_infuse_layer in infuse_layer:
                statement_merge=merge_lora.state_dict()
                ffn1.append(statement_merge[f'decoder.block.{i_infuse_layer}.layer.2.DenseReluDense.wi_0.weight'])
                ffn2.append(statement_merge[f"decoder.block.{i_infuse_layer}.layer.2.DenseReluDense.wi_1.weight"])
                ffn3.append(statement_merge[f'decoder.block.{i_infuse_layer}.layer.2.DenseReluDense.wo.weight'])
            tensor_dict={'ffn1':ffn1,'ffn2':ffn2,'ffn3':ffn3}
            torch.save(tensor_dict, 'weight_file/ours/tensor_lists.pt')
            exit()
        
        
        self.cpnet_vocab_size = n_concept
            
    
    def unfrozen_lora(self,module):
        TextKGMessagePassing_ours=module.base_model.model

        if self.model_type=='T5':
            for name,p in TextKGMessagePassing_ours.named_parameters():
                if 'encoder' in name or 'decoder' in name or 'lm_head' in name or 'shared' in name:
                    continue
                p.requires_grad = True
        elif self.model_type=='llama':
            for name,p in TextKGMessagePassing_ours.named_parameters():
                p.requires_grad = True
            freeze_net(TextKGMessagePassing_ours.model)
            freeze_net(TextKGMessagePassing_ours.lm_head)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    def generate(self, inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj, emb_data=None,subgraph_indice=None,**kwargs):
        if self.model_type=='llama':
            input_ids,attention_mask=inputs
        else:
            encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask = inputs

        # GNN inputs     

        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2 
        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        
        if self.context_embedding_zero and self.k>0:
            gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_type_ids.device)     

            gnn_input[:, 0] = 0      
            gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)
        else :
            gnn_input = None
        if self.k>0 and self.use_edge_score and self.context_embedding_zero: 

            q_type_id=0

            num_instance=gnn_input.size(0)
            question_entity_emb=[]
            for i in range(num_instance):
                question_temp_bool=gnn_input[i][node_type_ids[i]==q_type_id]
                if question_temp_bool.numel() == 0:
                    question_entity_emb.append(torch.ones(subgraph_indice[i],gnn_input.size(-1)).to(gnn_input.device))
                else:
                    question_entity_emb.append(question_temp_bool.mean(0,keepdim=True).repeat(subgraph_indice[i],1))
            assert len(question_entity_emb)==num_instance
            assert question_entity_emb[0].shape==(subgraph_indice[0],gnn_input.size(-1))
            question_entity_tensor=torch.cat(question_entity_emb,dim=0)   #(E,200)

            gnn_input_flat=gnn_input.view(-1,gnn_input.size(-1))
            

            edge_index, edge_type = adj   
            head_emb,tail_emb=gnn_input_flat.index_select(0,edge_index[0]),gnn_input_flat.index_select(0,edge_index[1])
            rel_emb=torch.cat((head_emb,tail_emb),dim=-1)    
            assert rel_emb.shape[0]==question_entity_tensor.shape[0]
            assert rel_emb.shape[1]==question_entity_tensor.shape[1]*2
            edge_score=self.similarity_compute_(question_entity_tensor,rel_emb) 
            if self.context_q_a_link_strong:
                batch_range=gnn_input.size(1)*torch.arange(gnn_input.size(0)).to(gnn_input.device)
                indice_context_head=torch.where(torch.eq(edge_index[0].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                indice_context_tail=torch.where(torch.eq(edge_index[1].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                
                

                indice_context=torch.unique(torch.cat((indice_context_head,indice_context_tail))).detach()
                edge_score_=edge_score.clone().to(gnn_input.device)
                
                edge_score_[indice_context]=torch.tensor([1.0,0.0]).to(gnn_input.device)
                
                edge_score=edge_score_
            
        else:
            edge_score=None

        if self.model_type=='T5':
            inputs_embeds=self.mp.shared(encoder_input_ids)
            result = self.mp.generate(input_ids=encoder_input_ids,attention_mask=encoder_attention_mask,
                                    #   decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,
                                      H=gnn_input, A=adj, node_type=node_type_ids, node_score=node_scores, 
                                        output_hidden_states=False,edge_score=edge_score,subgraph_indice=subgraph_indice,
                                        concept_ids=concept_ids,emb_data=emb_data,inputs_embeds_=inputs_embeds,**kwargs)
        else:
            inputs_embeds=None

            result = self.mp.generate(input_ids=input_ids, attention_mask=attention_mask,H=gnn_input, A=adj, node_type=node_type_ids, node_score=node_scores, 
                                        output_hidden_states=True,edge_score=edge_score,subgraph_indice=subgraph_indice,
                                        concept_ids=concept_ids,emb_data=emb_data,inputs_embeds=inputs_embeds,**kwargs)
        
        return result
        
    def forward(self, inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj, emb_data=None,subgraph_indice=None):

        #LM inputs
        if self.model_type=='T5':
            encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask = inputs
        else:
            input_ids, attention_mask = inputs
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)


            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
        # GNN inputs     

        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2    #context node的id变为cpnet_vocab_size + 2
        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        
        if self.context_embedding_zero and self.k >0:
            gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_type_ids.device)    

            gnn_input[:, 0] = 0       
            gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)
        else :
            gnn_input = None
            
            
            
        if self.k>0 and self.use_edge_score and self.context_embedding_zero: 

            q_type_id=0

            num_instance=gnn_input.size(0)
            question_entity_emb=[]
            for i in range(num_instance):
                question_temp_bool=gnn_input[i][node_type_ids[i]==q_type_id]
                if question_temp_bool.numel() == 0:
                    question_entity_emb.append(torch.ones(subgraph_indice[i],gnn_input.size(-1)).to(gnn_input.device))
                else:
                    question_entity_emb.append(question_temp_bool.mean(0,keepdim=True).repeat(subgraph_indice[i],1))
            assert len(question_entity_emb)==num_instance
            assert question_entity_emb[0].shape==(subgraph_indice[0],gnn_input.size(-1))
            question_entity_tensor=torch.cat(question_entity_emb,dim=0)   #(E,200)

            gnn_input_flat=gnn_input.view(-1,gnn_input.size(-1))
            

            edge_index, edge_type = adj  
            head_emb,tail_emb=gnn_input_flat.index_select(0,edge_index[0]),gnn_input_flat.index_select(0,edge_index[1])
            rel_emb=torch.cat((head_emb,tail_emb),dim=-1)    
            assert rel_emb.shape[0]==question_entity_tensor.shape[0]
            assert rel_emb.shape[1]==question_entity_tensor.shape[1]*2
            edge_score=self.similarity_compute_(question_entity_tensor,rel_emb)  
            if self.context_q_a_link_strong:
                batch_range=gnn_input.size(1)*torch.arange(gnn_input.size(0)).to(gnn_input.device)
                indice_context_head=torch.where(torch.eq(edge_index[0].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                indice_context_tail=torch.where(torch.eq(edge_index[1].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                
                
                indice_context=torch.unique(torch.cat((indice_context_head,indice_context_tail))).detach()
                edge_score_=edge_score.clone().to(gnn_input.device)
                
                edge_score_[indice_context]=torch.tensor([1.0,0.0]).to(gnn_input.device)
                
                edge_score=edge_score_
            
        else:
            edge_score=None
        
        if self.model_type=='llama':
            output= self.mp(input_ids=input_ids, attention_mask=attention_mask, H=gnn_input, A=adj, node_type=node_type_ids,node_score= node_scores, 
                                        edge_score=edge_score,subgraph_indice=subgraph_indice,
                                        concept_ids=concept_ids,emb_data=emb_data,position_ids=position_ids,return_dict=True,output_attentions=False,
                                        output_hidden_states=False,past_key_values=None,use_cache=False)
        else:
            output= self.mp(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, H=gnn_input, A=adj, node_type=node_type_ids, node_score=node_scores,output_hidden_states=False, 
                                        edge_score=edge_score,subgraph_indice=subgraph_indice,
                                        concept_ids=concept_ids,emb_data=emb_data,return_dict=True,output_attentions=False,
                                        past_key_values=None,use_cache=False,decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask)
        return output.logits



class TextKGMessagePassing(ModelClass):
    def __init__(self, config, args={}, k=5, n_ntype=4, n_etype=38, dropout=0.2, concept_dim=200, 
                 p_fc=0.2,infuse_layer=-1,
                 num_key_value=1,question_rel_similarity='BiLinearSimilarity',
                 use_edge_score=False,edge_class='hard',context_q_a_link_strong=True,context_embedding_zero=True,n_concept=799273,concept_in_dim=1024,pretrained_concept_emb=None, 
                 freeze_ent_emb=True,p_emb=0.02,frozen_lm=False,model_type='llama',unfreeze_infuse=False,train_header=False,gradient_checkpointing=False):
        super().__init__(config=config)

        self.gradient_checkpointing=gradient_checkpointing
        if self.gradient_checkpointing:

            self.model.gradient_checkpointing_enable()

        if frozen_lm:
            if ModelClass==LlamaForCausalLM:
                freeze_net(self.model)
                freeze_net(self.lm_head)
            else:
                raise ValueError('ModelClass must be one of LlamaModel,LlamaForCausalLM')

        self.sent_dim = config.hidden_size
        self.dropout_e = nn.Dropout(p_emb)
        self.context_embedding_zero=context_embedding_zero
        self.use_edge_score=use_edge_score
        self.context_q_a_link_strong=context_q_a_link_strong
        if not self.context_embedding_zero and self.use_edge_score:
            self.similarity_compute_=layers.similarity_compute(question_rel_similarity,edge_class,concept_dim)
        if not self.context_embedding_zero:
            self.svec2nvec = nn.Linear(self.sent_dim, concept_dim)
        if k >= 0 and not self.context_embedding_zero:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim)
        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k
        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)
        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.num_key_value=num_key_value
        self.infuse_layer = infuse_layer

        infuse_layer_gnn=infuse_layer[-k:]
        self.infuse_key_up_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.infuse_key_gate_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.infuse_value_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.num_hidden_layers = config.num_hidden_layers

        if unfreeze_infuse:
            for infuse_i in self.infuse_layer:

                # if infuse_i>=self.num_hidden_layers - self.k:
                if True:
                    unfreeze_net(self.model.layers[infuse_i])
        
        for key in self.infuse_value_modules.keys():
            init.constant_(self.infuse_value_modules[key].weight, 0)
            init.constant_(self.infuse_value_modules[key].bias, 0)
            # self.infuse_value_modules[key].weight.data.zero_()
            # self.infuse_value_modules[key].bias.data.zero_()
            

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, concept_dim), torch.nn.BatchNorm1d(concept_dim), torch.nn.ReLU(), torch.nn.Linear(concept_dim, concept_dim))
        self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(concept_dim, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.concept_dim = concept_dim
        

    def LlamaGAT(self, inputs_embeds, attention_mask, position_ids, _X, edge_index, edge_type, 
                _node_type, _node_feature_extra ,output_attentions=None, output_hidden_states=None,edge_score=None,
                past_key_values=None,use_cache=None,return_dict=None):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        position_ids: 
        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        batch_size,seq_length = inputs_embeds.size(0),inputs_embeds.size(1)

        # past_seen_tokens = 0
        
        seq_length_with_past=seq_length
        
        past_key_values_length=0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
        if position_ids is None:
            device = inputs_embeds.device 
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        if ModelClass==LlamaForCausalLM:
            attention_mask = self.model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        else:
            raise ValueError('ModelClass must be one of LlamaForCausalLM')          
            

        hidden_states=inputs_embeds
            
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None  
            
            
        bs = hidden_states.size(0)
        for i, layer_module in enumerate(self.model.layers):
            key_gate,key_up,value=None,None,None
            #gnn
            #注入后面几层
            gnn_flag=False
            if i >= self.num_hidden_layers - self.k:
                gnn_layer_index = i - self.num_hidden_layers + self.k
                gnn_flag=True



            if gnn_flag:

                

                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra,edge_score)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                if i in self.infuse_layer:

                    X = _X.view(bs, -1, _X.size(1))[:,self.num_key_value]

                    key_gate=self.infuse_key_gate_modules[str(i)](X)
                    key_up=self.infuse_key_up_modules[str(i)](X)
                    value=self.infuse_value_modules[str(i)](X)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                layer_outputs=checkpoint(layer_module,hidden_states, attention_mask,position_ids,past_key_values,
                                         output_attentions,use_cache,(key_gate,key_up),value,use_reentrant=False)

            else:
                # print('not checkpoint')
                layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask,position_ids=position_ids,past_key_value=past_key_value,
                                         output_attentions=output_attentions,use_cache=use_cache,key=(key_gate,key_up),value=value)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if ModelClass==LlamaForCausalLM:
            hidden_states = self.model.norm(hidden_states)
        else:
            raise ValueError('ModelClass must be one of LlamaForCausalLM')
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            # gnn_output=_X
        ) 
    def compute_edge_score(self,context_embedding,concept_ids,emb_data,node_type,subgraph_indice,A):
        if not self.context_embedding_zero:
            gnn_input0 = self.activation(self.svec2nvec(context_embedding)).unsqueeze(1) #(batch_size, 1, dim_node)
            gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
            gnn_input1 = gnn_input1.to(node_type.device)    
            gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)
            H=gnn_input
        if self.use_edge_score and not self.context_embedding_zero: 
            q_type_id=0
       

            num_instance=gnn_input.size(0)
            question_entity_emb=[]
            for i in range(num_instance):
                question_temp_bool=gnn_input[i][node_type[i]==q_type_id]
                if question_temp_bool.numel() == 0:
                    question_entity_emb.append(torch.ones(subgraph_indice[i],gnn_input.size(-1)).to(gnn_input.device))
                else:
                    question_entity_emb.append(question_temp_bool.mean(0,keepdim=True).repeat(subgraph_indice[i],1))
            assert len(question_entity_emb)==num_instance
            assert question_entity_emb[0].shape==(subgraph_indice[0],gnn_input.size(-1))
            question_entity_tensor=torch.cat(question_entity_emb,dim=0)   #(E,200)

            gnn_input_flat=gnn_input.view(-1,gnn_input.size(-1))

            edge_index, edge_type = A   
            head_emb,tail_emb=gnn_input_flat.index_select(0,edge_index[0]),gnn_input_flat.index_select(0,edge_index[1])
            rel_emb=torch.cat((head_emb,tail_emb),dim=-1)    
            assert rel_emb.shape[0]==question_entity_tensor.shape[0]
            assert rel_emb.shape[1]==question_entity_tensor.shape[1]*2

            edge_score=self.similarity_compute_(question_entity_tensor,rel_emb)   #[E,2]，如果是这样的话，我们规定第一个值表示边存在的概率，第二个值表示不存在的概率
            if self.context_q_a_link_strong:
                batch_range=gnn_input.size(1)*torch.arange(gnn_input.size(0)).to(gnn_input.device)
                indice_context_head=torch.where(torch.eq(edge_index[0].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                indice_context_tail=torch.where(torch.eq(edge_index[1].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                
                

                indice_context=torch.unique(torch.cat((indice_context_head,indice_context_tail))).detach()
                edge_score_=edge_score.clone().to(gnn_input.device)
                
                edge_score_[indice_context]=torch.tensor([1.0,0.0]).to(gnn_input.device)
                
                edge_score=edge_score_
        else:
            edge_score=None
        return edge_score,H

    def forward(self, labels=None,input_ids=None, attention_mask=None, H=None, A=None, node_type=None, node_score=None, output_hidden_states=True,edge_score=None, 
                subgraph_indice=None,concept_ids=None,emb_data=None, position_ids=None,return_dict=True,output_attentions=False,
                past_key_values=None,use_cache=False,inputs_embeds=None,task_ids=None):

        # LM inputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        seq_len = input_ids.size(1)
        if ModelClass==LlamaForCausalLM:
            inputs_embeds=self.model.embed_tokens(input_ids)
        else:
            raise ValueError('ModelClass must be one of LlamaModel,LlamaForCausalLM')
        

        if self.k>0 and not self.context_embedding_zero and self.use_edge_score:
            edge_score,H=self.compute_edge_score(context_embedding=inputs_embeds[:,0],concept_ids=concept_ids,emb_data=emb_data,node_type=node_type,subgraph_indice=subgraph_indice,A=A)
            


        #Embed type
        T = modeling_gnn.make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        if self.k>0 :
            X = H
            edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
            _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        else:
            _X=None
            edge_index=None
            edge_type=None
        
        
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]

        _node_feature_extra = node_type_emb.view(_node_type.size(0), -1).contiguous()

        outputs = self.LlamaGAT(inputs_embeds,attention_mask, position_ids, _X, edge_index, edge_type, _node_type, _node_feature_extra,output_hidden_states=output_hidden_states,edge_score=edge_score,
                                output_attentions=output_attentions,past_key_values=past_key_values,use_cache=use_cache,return_dict=return_dict)
        #logits
        hidden_states=outputs[0]
        if self.gradient_checkpointing:
            logits=checkpoint(self.lm_head,hidden_states)
        else:
            logits=self.lm_head(hidden_states)

        
        loss=None
        if not return_dict:
            output=(logits,)+outputs[1:]
            return (loss,)+output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss,
                                      logits=logits,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions,
                                    #   gnn_output=gnn_output,
                                      )                          
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "H":kwargs.get("H"),
                "A":kwargs.get("A"),
                "node_type":kwargs.get("node_type"),
                "node_score":kwargs.get("node_score"),
                "subgraph_indice":kwargs.get("subgraph_indice"),
                "concept_ids":kwargs.get("concept_ids"),
                "emb_data":kwargs.get("emb_data"),
                "edge_score":kwargs.get("edge_score"),
            }
        )
        return model_inputs
    
    
    

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "H":kwargs.get("H"),
                "A":kwargs.get("A"),
                "node_type":kwargs.get("node_type"),
                "node_score":kwargs.get("node_score"),
                "subgraph_indice":kwargs.get("subgraph_indice"),
                "concept_ids":kwargs.get("concept_ids"),
                "emb_data":kwargs.get("emb_data"),
                "edge_score":kwargs.get("edge_score"),
            }
        )
        return model_inputs

class TextKGMessagePassing_T5(T5ForConditionalGeneration):
    def __init__(self, config, args={}, k=5, n_ntype=4, n_etype=38, dropout=0.2, concept_dim=200, 
                  p_fc=0.2, infuse_layer=-1,
                 num_key_value=1,question_rel_similarity='BiLinearSimilarity',
                 use_edge_score=False,edge_class='hard',context_q_a_link_strong=True,context_embedding_zero=True,n_concept=799273,concept_in_dim=1024,pretrained_concept_emb=None, 
                 freeze_ent_emb=True,p_emb=0.02,frozen_lm=False,model_type='llama',unfreeze_infuse=False,train_header=False,gradient_checkpointing=False,
                 ):
        super().__init__(config=config)
        self.gradient_checkpointing=gradient_checkpointing
        if self.gradient_checkpointing:
            # self.model.gradient_checkpointing_enable({"key":None,"value":None})
            self.model.gradient_checkpointing_enable()
        # if not train_header:
        #     freeze_net(self.lm_head)
            # freeze_net(self.model.embed_tokens)
        if frozen_lm:
            freeze_net(self.shared)
            freeze_net(self.lm_head)
            freeze_net(self.encoder)
            freeze_net(self.decoder)


        self.sent_dim = config.hidden_size
        self.dropout_e = nn.Dropout(p_emb)
        self.context_embedding_zero=context_embedding_zero
        self.use_edge_score=use_edge_score
        self.context_q_a_link_strong=context_q_a_link_strong
        if not self.context_embedding_zero and self.use_edge_score:
            self.similarity_compute_=layers.similarity_compute(question_rel_similarity,edge_class,concept_dim)
        if not self.context_embedding_zero:
            self.svec2nvec = nn.Linear(self.sent_dim, concept_dim)
        if k >= 0 and not self.context_embedding_zero:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.hidden_size = concept_dim

        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim)
        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k
        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)
        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.num_key_value=num_key_value
        self.infuse_layer = infuse_layer

        infuse_layer_gnn=infuse_layer[-k:]

        self.infuse_key_up_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.infuse_key_gate_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.infuse_value_modules=nn.ModuleDict({str(infuse_layer_gnn[i]):nn.Linear(concept_dim, config.hidden_size) for i in range(len(infuse_layer_gnn))})
        self.num_hidden_layers = config.num_hidden_layers
        if unfreeze_infuse:
            for infuse_i in self.infuse_layer:
                unfreeze_net(self.decoder.block[infuse_i])
        for key in self.infuse_value_modules.keys():
            init.constant_(self.infuse_value_modules[key].weight, 0)
            init.constant_(self.infuse_value_modules[key].bias, 0)
            # self.infuse_value_modules[key].weight.data.zero_()
            # self.infuse_value_modules[key].bias.data.zero_()
            
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, concept_dim), torch.nn.BatchNorm1d(concept_dim), torch.nn.ReLU(), torch.nn.Linear(concept_dim, concept_dim))
        self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(concept_dim, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])



        self.concept_dim = concept_dim
        
    def t5gat_decoder(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,_X=None, edge_index=None, edge_type=None, _node_type=None,
        _node_feature_extra=None ,edge_score=None):

        if self.decoder.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
            
        output_attentions = output_attentions if output_attentions is not None else self.decoder.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.decoder.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.decoder.config.use_cache
        return_dict = return_dict if return_dict is not None else self.decoder.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" 
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" 
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
 
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)

        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length



        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if encoder_attention_mask is None and encoder_hidden_states is not None:      
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.decoder.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.decoder.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.decoder.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.decoder.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        head_mask = self.decoder.get_head_mask(head_mask, self.decoder.config.num_layers)
        cross_attn_head_mask = self.decoder.get_head_mask(cross_attn_head_mask, self.decoder.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.decoder.dropout(inputs_embeds)
        bs=hidden_states.size(0)
        
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.decoder.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]  

            if self.decoder.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            key_gate,key_up,value=None,None,None
           
            if i >= self.num_hidden_layers - self.k:
                
                
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k

                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra,edge_score)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                if i in self.infuse_layer:
                    X = _X.view(bs, -1, _X.size(1))[:,self.num_key_value]
                    
                    key_gate=self.infuse_key_gate_modules[str(i)](X)
                    key_up=self.infuse_key_up_modules[str(i)](X)
                    value=self.infuse_value_modules[str(i)](X)
            if output_hidden_states:  
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.decoder.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    key=(key_gate,key_up),
                    value=value
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
                
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if  encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
            if self.decoder.model_parallel:
                for k, v in self.decoder.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.decoder.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.decoder.final_layer_norm(hidden_states)
        hidden_states = self.decoder.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        else:
                
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )        


    def t5GAT(self, input_ids,inputs_embeds, attention_mask, _X, edge_index, edge_type, 
                _node_type, _node_feature_extra ,output_attentions=None, output_hidden_states=None,edge_score=None,
                past_key_values=None,use_cache=None,return_dict=None,decoder_input_ids=None,decoder_attention_mask=None,head_mask=None,
                decoder_head_mask=None,cross_attn_head_mask=None,encoder_outputs=None,decoder_inputs_embeds=None,labels=None):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        position_ids: 
        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        bs=inputs_embeds.size(0)
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:   
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:    
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):  
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:    
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # encoder_hidden_states=hidden_states
        
        decoder_outputs = self.t5gat_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            _X=_X, 
            edge_index=edge_index, 
            edge_type=edge_type, 
            _node_type=_node_type,
            _node_feature_extra=_node_feature_extra,
            edge_score=edge_score,
        )
        

        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
            
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.gradient_checkpointing and self.training:
            lm_logits = checkpoint(self.lm_head,sequence_output)
        else:
            lm_logits=self.lm_head(sequence_output)
        loss = None
        # if labels is not None:
        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        #     # move labels to correct device to enable PP
        #     labels = labels.to(lm_logits.device)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



    def compute_edge_score(self,context_embedding,concept_ids,emb_data,node_type,subgraph_indice,A):
        if not self.context_embedding_zero:
            gnn_input0 = self.activation(self.svec2nvec(context_embedding)).unsqueeze(1) #(batch_size, 1, dim_node)
            gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
            gnn_input1 = gnn_input1.to(node_type.device)    
            gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)
            H=gnn_input
        if self.use_edge_score and not self.context_embedding_zero: 
            q_type_id=0

            num_instance=gnn_input.size(0)
            question_entity_emb=[]
            for i in range(num_instance):
                question_temp_bool=gnn_input[i][node_type[i]==q_type_id]
                if question_temp_bool.numel() == 0:
                    question_entity_emb.append(torch.ones(subgraph_indice[i],gnn_input.size(-1)).to(gnn_input.device))
                else:
                    question_entity_emb.append(question_temp_bool.mean(0,keepdim=True).repeat(subgraph_indice[i],1))
            assert len(question_entity_emb)==num_instance
            assert question_entity_emb[0].shape==(subgraph_indice[0],gnn_input.size(-1))
            question_entity_tensor=torch.cat(question_entity_emb,dim=0)   #(E,200)

            gnn_input_flat=gnn_input.view(-1,gnn_input.size(-1))

            edge_index, edge_type = A  
            head_emb,tail_emb=gnn_input_flat.index_select(0,edge_index[0]),gnn_input_flat.index_select(0,edge_index[1])
            rel_emb=torch.cat((head_emb,tail_emb),dim=-1)    
            assert rel_emb.shape[0]==question_entity_tensor.shape[0]
            assert rel_emb.shape[1]==question_entity_tensor.shape[1]*2

            edge_score=self.similarity_compute_(question_entity_tensor,rel_emb)  
            if self.context_q_a_link_strong:
                batch_range=gnn_input.size(1)*torch.arange(gnn_input.size(0)).to(gnn_input.device)
                indice_context_head=torch.where(torch.eq(edge_index[0].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                indice_context_tail=torch.where(torch.eq(edge_index[1].unsqueeze(1),batch_range.unsqueeze(0)))[0]
                

                indice_context=torch.unique(torch.cat((indice_context_head,indice_context_tail))).detach()
                edge_score_=edge_score.clone().to(gnn_input.device)
                
                edge_score_[indice_context]=torch.tensor([1.0,0.0]).to(gnn_input.device)
                
                edge_score=edge_score_
        else:
            edge_score=None
        return edge_score,H


    def forward(self, labels=None,input_ids=None, attention_mask=None, H=None, A=None, node_type=None, node_score=None, output_hidden_states=True,edge_score=None, 
                subgraph_indice=None,concept_ids=None,emb_data=None,return_dict=True,output_attentions=False,
                past_key_values=None,use_cache=False,decoder_input_ids=None,decoder_attention_mask=None,head_mask=None,
                decoder_head_mask=None,cross_attn_head_mask=None,encoder_outputs=None,inputs_embeds=None,inputs_embeds_=None,decoder_inputs_embeds=None):


        _batch_size, _n_nodes = node_type.size()



        if inputs_embeds_ is None:
            if input_ids is not None:
                inputs_embeds_=self.shared(input_ids)
            else:
                raise ValueError('input_ids is None')
        if self.k>0 and not self.context_embedding_zero and self.use_edge_score:
            edge_score,H=self.compute_edge_score(context_embedding=inputs_embeds_[:,0],concept_ids=concept_ids,emb_data=emb_data,node_type=node_type,subgraph_indice=subgraph_indice,A=A)
            


        #Embed type
        T = modeling_gnn.make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        if self.k>0 :
            X = H
            edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
            _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        else:
            _X=None
            edge_index=None
            edge_type=None
        
        
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = node_type_emb.view(_node_type.size(0), -1).contiguous()

        outputs = self.t5GAT(input_ids,inputs_embeds_,attention_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra,
                             output_hidden_states=output_hidden_states,edge_score=edge_score,
                                output_attentions=output_attentions,past_key_values=past_key_values,use_cache=use_cache,
                                return_dict=return_dict,decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,cross_attn_head_mask=cross_attn_head_mask,encoder_outputs=encoder_outputs,decoder_inputs_embeds=decoder_inputs_embeds)
        
        return outputs

        
             
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
            
            
        return {
            ""
            # "decoder_attention_mask":attention_mask,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "H":kwargs.get("H"),
            "A":kwargs.get("A"),
            "node_type":kwargs.get("node_type"),
            "node_score":kwargs.get("node_score"),
            "subgraph_indice":kwargs.get("subgraph_indice"),
            "concept_ids":kwargs.get("concept_ids"),
            "emb_data":kwargs.get("emb_data"),
            "edge_score":kwargs.get("edge_score"),
            "inputs_embeds_":kwargs.get("inputs_embeds_"),
        }

