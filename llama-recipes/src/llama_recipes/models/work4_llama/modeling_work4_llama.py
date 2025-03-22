from typing import List, Optional, Tuple, Union
from ..llama.modeling_llama import LlamaForCausalLM
from ..llama.modeling_llama import LlamaDecoderLayer
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import os
from peft import get_peft_model, prepare_model_for_kbit_training
from ...utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    generate_peft_config,
)
# from ...datasets.work4_ctrg_dataset import get_seetings
from ...configs.training import get_seetings

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
import copy
import os
import math
import numpy as np


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def attention_know(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (b, 8, 33, 33)
    if mask is not None:
        # print(mask.size())  torch.Size([b, 1, 33, 33])
        scores = torch.mul(scores, mask)
        # print(scores)
        # print(scores.size())  torch.Size([8, 4, 33, 33])
        scores = scores.masked_fill(mask == 0, -1e9)
        # print(scores)
        # print(scores.size())  torch.Size([8, 4, 33, 33])
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention_know(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention_know, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention_know(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Grapher(nn.Module):
    def __init__(self, layer, N):
        super(Grapher, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, enc_output, src_mask=None):
        # batch = enc_output.size(0)
        # graph_mask = knowledge_graph
        # graph_feature = self.graph_model(batch, graph_mask, self.graph_nodes)
        entity_embeddings = enc_output
        for layer in self.layers:
            entity_embeddings = layer(entity_embeddings, src_mask)
        return self.norm(entity_embeddings)


class GrapherLayer(nn.Module):
    def __init__(self, d_model, graphattn, feed_forward, dropout):
        super(GrapherLayer, self).__init__()
        self.graphattn = graphattn
        # self.visual_attn = visualattn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, entity_embeddings, src_mask=None):
        x = self.sublayer[0](entity_embeddings, lambda t: self.graphattn(t, entity_embeddings, entity_embeddings, src_mask))  # 16,28,512
        graph_feat = self.sublayer[1](x, self.feed_forward)
        return graph_feat  # b, 33, 512

class Classfier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(Classfier, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.cls_head = nn.Linear(33 * hidden_dim, 32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[:, :, :]
        batch, node_number, input_dim = x.size(0), x.size(1), x.size(2)
        x = self.fc(x.view(-1, input_dim)).view(batch, node_number, self.hidden_dim)
        x = x.view(batch, -1)
        x = self.cls_head(x)  # b,32
        x = self.sigmoid(x)
        return x

class Pathology_Balanced_Prompting_Module(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.num_heads = 8
        self.dropout = 0.1
        self.d_model = 512
        self.d_ff = 512
        self.num_layers = 3

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        attn_know = MultiHeadedAttention_know(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        # Brain CT Medical Entities
        self.pathology_entity_Chinese = ['[global_node]', '侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室', '半卵圆中心', '枕叶',
                                         '中线', '第四脑室',
                                         '脑沟', '颞枕叶', '额叶', '基底节区', '脑实质', '筛窦', '颞叶', '蝶窦', '丘脑',
                                         '高密度影', '低密度影',
                                         '减低', "受压", '水肿带', '增宽', '增厚', '增高', '密度影', '变窄', '左移',
                                         '移位', '肿胀']
        self.pathology_entity_English = ['[global_node]', 'lateral ventricle', 'brainstem', 'maxillary sinus', 'third ventricle',
                                         'pariental lobe',
                                         'ventricle', 'half oval center', 'occipital lobe', 'midline structure',
                                         'fourth ventricle', 'sulcus cerebri',
                                         'temporal occipital lobe', 'frontal lobe', 'basal ganglia',
                                         'cerebral parenchyma', 'ethmoid sinus',
                                         'temporal lobe', 'sphenoid sinus', 'thalamus', 'high density', 'low density',
                                         'decrease', 'compressed',
                                         'edema', 'widen', 'thickening', 'increase', 'density', 'narrow', 'shift left',
                                         'displacement', 'swelling']
        self.pathology_entity_Chinese_half = ['[global_node]', '侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室',
                                         '半卵圆中心', '枕叶', '中线',
                                         '高密度影', '低密度影',
                                         '减低', "受压", '水肿带', '增宽', '增厚']

        self.Visaul_Mapper = nn.Sequential(nn.Linear(2048, 512))
        self.Entity_Mapper = nn.Sequential(nn.Linear(4096, 512))
        self.Cross_Attention = c(attn)
        self.Knowledge_Projector = nn.Linear(512, 512)
        self.Knowledge_Predicter = nn.Sequential(nn.Linear(33*33, 33*33), nn.Sigmoid())
        self.Knowledge_norm = nn.Sigmoid()
        # Knowledge-driven Joint Attention
        self.KoJo_Att = Grapher(GrapherLayer(self.d_model, c(attn_know), c(ff), self.dropout), self.num_layers)
        # Pathology Adaptor
        self.Pathology_Adaptor = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 4096))
        # Self-Adaptive Pathology Scorer
        self.Pathology_Scorer = Classfier(self.d_model, 128, self.dropout)
    

    def forward(self, visual_inputs, entity_inputs, pathological_graph, batch_idxs=None):
        # 1.Mapper
        batch_size = visual_inputs.size(0)
        visual_inputs = self.Visaul_Mapper(visual_inputs.view(-1, 2048))  # b*24, 512
        entity_inputs = self.Entity_Mapper(entity_inputs.view(-1, 4096))  # b*33, 512
        visual_inputs, entity_inputs = visual_inputs.view(batch_size, 24, 512), entity_inputs.view(batch_size, 33, 512)  # b, 24, 512   b, 33, 512
        # 2.Cross Att
        entity_features = self.Cross_Attention(entity_inputs, visual_inputs, visual_inputs, None)  # 6, 33, 512
        # 3. predict relations
        entity_relations = entity_features.clone()  # 6, 33, 512
        entity_relations = self.Knowledge_Projector(entity_relations)  # 6, 33, 512
        entity_relations = torch.matmul(entity_relations.clone(), entity_relations.clone().permute(0, 2, 1)).cuda()  # 6, 33, 512 * 6, 512, 33 -> 6, 33, 33
        Empirical_Adjaceny_Matrix = self.Knowledge_Predicter(entity_relations.view(batch_size, -1)).view(-1, 33, 33)  # 6, 33, 33
        # 4. merge adjaceny_Matrix
        a_E, a_I = 0.9, 0.1
        Adjaceny_Matrix = a_E * pathological_graph + a_I * Empirical_Adjaceny_Matrix  # 6, 33, 33
        # 5. Knowledge-driven Joint Attention
        entity_embeddings = self.KoJo_Att(entity_features, Adjaceny_Matrix)  # 6, 33, 512
        # 6. Pathology Adaptor
        adapted_entity_embeddings = self.Pathology_Adaptor(entity_embeddings)  # 6, 33, 4096
        # 7. Self-Adaptive Pathology Scorer
        cls_res = self.Pathology_Scorer(entity_embeddings)
        return adapted_entity_embeddings, cls_res


class Multi_Modal_Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_feature_extraction = nn.Linear(49*2048, 2048)
        self.global_feature_mapping = nn.Linear(2048, 4096)

    def forward(self, visual_inputs):
        batch = visual_inputs.size(0)
        att_feats = visual_inputs.view(batch, 24, 14, 14, 2048)  # (batch ,24, 196, 2048)
        feats = F.avg_pool3d(att_feats.permute(0, 1, 4, 2, 3), kernel_size=(1, 2, 2)).permute(0, 1, 3, 4, 2)  # torch.Size([batch, 24, 7, 7, 2048])
        feats = feats.reshape(batch, 24, -1, 2048)  # torch.Size([batch, 24, 49, 2048])
        feats = self.global_feature_extraction(feats.reshape(batch, 24, -1))  # torch.Size([batch, 24, 2048])
        visual_embeddings = self.global_feature_mapping(feats.reshape(batch, 24, -1))  # torch.Size([batch, 24, 2048])

        return visual_embeddings


class ResNetLlamaModel(nn.Module):
    '''
    A simple vision-language model, inspired by the LLaVA architecture.
    It accepts 2048-dimensional visual features as input (these features have already been extracted by ResNet101 and stored in a file).
    An MLP is used as a Projector to project these visual features to 4096 dimensions, which then serve as visual special tokens.
    The language model used here is Llama3, quantized using int4.
    '''

    def __init__(self, train_config, use_cache, tokenizer, image_token_index, kwargs, wandb_run):
        super().__init__()

        self.tokenizer = tokenizer

        self.image_token_index = image_token_index
        self.pathological_token_id = int(tokenizer.encode("<|pathology_feature|>")[-1])
        self.status_token_id = int(tokenizer.encode("<|pathology_status|>")[-1])
        self.cls_token_id = int(tokenizer.encode("<|pathology_cls|>")[-1])
        
        self.ignore_index = -100
        self.stop_token_id = int(tokenizer.encode("<|end_of_text|>")[-1])  # 128001
        self.pad_token_id = self.stop_token_id
        self.context_length = train_config.context_length
        self.save_peft_model_name = "peft_model_lora_adapter.pth"

        # Visual Adaptor
        self.multi_modal_projector = Multi_Modal_Projector()
        

        self.language_model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_4bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )
        # If there is a mismatch between tokenizer vocab size and embedding matrix,
        # throw a warning and then expand the embedding matrix
        if len(tokenizer) > self.language_model.get_input_embeddings().weight.shape[0]:
            print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            self.language_model.resize_token_embeddings(len(tokenizer))
        # print_model_size(self.text_model, train_config, rank if train_config.enable_fsdp else 0)
        # Prepare the model for int8 training if quantization is enabled
        if train_config.quantization:
            self.language_model = prepare_model_for_kbit_training(self.language_model)
        # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
        if train_config.use_peft:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(self.language_model, peft_config)
            model.print_trainable_parameters()
            if wandb_run:
                wandb_run.config.update(peft_config)
        self.Pathology_Balanced_Prompting_Module = Pathology_Balanced_Prompting_Module(self.tokenizer)
        self.pathology_entity_Chinese = ['侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室',
                                         '半卵圆中心', '枕叶', '中线', '第四脑室',
                                         '脑沟', '颞枕叶', '额叶', '基底节区', '脑实质', '筛窦', '颞叶', '蝶窦', '丘脑',
                                         '高密度影', '低密度影', '减低', "受压", '水肿带', '增宽', '增厚', '增高', '密度影',
                                         '变窄', '左移', '移位', '肿胀']
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)  # 24
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(
            input_ids != self.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[
            batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )  # (b, text_len, 4096)
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )  # (b, text_len)
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )  # (b, text_len)
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]  # batch_indices tensor([0, 0, 0,  ..., 5, 5, 5], device='cuda:0')，  non_image_indices tensor([  0,   1,  26,  ..., 252, 253, 254], device='cuda:0')
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # (b, text_len)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(
            target_device)  # (b, text_len)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():  # b*24 = 24
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(
            target_device)  # final_embedding torch.Size([6, 99, 4096]);  final_embedding[image_to_overwrite] torch.Size([144, 4096])
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _merge_input_ids_with_pathological_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = (input_ids == self.image_token_index) | (input_ids == self.pathological_token_id)  # <|pathology_feature|> True  128262
        # special_image_token_mask = input_ids == self.pathological_token_id or input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)  # 24 + 32
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(
            (input_ids != self.image_token_index) & (input_ids != self.pathological_token_id) )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:  # 不走这里
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[
            batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )  # (b, text_len, 4096)
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )  # (b, text_len)
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )  # (b, text_len)
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device. 加载到GPU上
        target_device = inputs_embeds.device  
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]  # 记录文本编码的位置，把attention_mask中文本编码放进来，特殊字符是0，后面多余的终止符也是0
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]  # torch.int64

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # (b, text_len) 布尔值，文本位置是False，特殊字符位置是True
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(
            target_device)  # (b, text_len)
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():  # image_to_overwrite.sum()  206   image_features.shape[:-1].numel() 192
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)  # final_embedding torch.Size([6, 99, 4096]);  final_embedding[image_to_overwrite] torch.Size([144, 4096])
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        # batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        # indices_to_mask = new_token_positions[batch_indices, pad_indices]
        #
        # final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


    def _merge_input_ids_with_status_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.status_token_id  # <|pathology_status|> True  128263
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length  # = sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.status_token_id)  # batch_indices(b*text_len)记录了每个batch的信息; non_image_indices存储了当前text token中全部非<img>的位置

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[
            batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )  # (b, text_len, 4096)
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )  # (b, text_len)
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )  # (b, text_len)
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]  # torch.int64

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # (b, text_len)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(
            target_device)  # (b, text_len)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():  # image_to_overwrite.sum()  206   image_features.shape[:-1].numel() 192
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)  # final_embedding torch.Size([6, 99, 4096]);  final_embedding[image_to_overwrite] torch.Size([144, 4096])
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        # batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        # indices_to_mask = new_token_positions[batch_indices, pad_indices]
        #
        # final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids
    

    def _merge_input_ids_with_cls_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.cls_token_id  # <|pathology_cls|> True  128263
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)  # [32, 32, 32, ...]
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length  # = sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.cls_token_id)
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[
            batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )  # (b, text_len, 4096)
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )  # (b, text_len)
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )  # (b, text_len)
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]  # torch.int64

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # (b, text_len)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(
            target_device)  # (b, text_len)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():  # image_to_overwrite.sum()  206   image_features.shape[:-1].numel() 192
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)  # final_embedding torch.Size([6, 99, 4096]);  final_embedding[image_to_overwrite] torch.Size([144, 4096])
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        # batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        # indices_to_mask = new_token_positions[batch_indices, pad_indices]
        #
        # final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


    def _status_promptor_batch_avg(self, predicted_labels, ground_labels):
        node_num = predicted_labels.size(1)
        node_course = torch.zeros(node_num)
        BCE_crition = torch.nn.BCELoss(reduction='none')
        for node_idx in range(node_num):
            predict_node = predicted_labels[:, node_idx].unsqueeze(1)
            target_node = ground_labels[:, node_idx].unsqueeze(1)
            BCEloss_node = BCE_crition(predict_node, target_node).sum(1)
            loss_node_cls = BCEloss_node.mean()
            node_course[node_idx] = loss_node_cls
        normalized_scores = 1 - torch.exp(-node_course * 1.0)
        status_words = ['exceptional ', 'proficient ', 'moderate', 'limited', 'inadequate']
        status_labels = [
            status_words[0] if score <= 0.2 else
            status_words[1] if score <= 0.4 else
            status_words[2] if score <= 0.6 else
            status_words[3] if score <= 0.8 else
            status_words[4] for score in normalized_scores
        ]
        batch_samples_info = status_labels
        batch_samples_tokens = [torch.tensor(self.tokenizer.encode(word)).cuda() for word in status_labels]

        return batch_samples_info, batch_samples_tokens
    

    def _status_promptor_for_generation_batch_avg(self, predicted_labels):
        batch_size = predicted_labels.size(0)
        node_num = predicted_labels.size(1) 
        status_words = ['exceptional ', 'proficient ', 'moderate', 'limited', 'inadequate']
        # use 'moderate' for inference
        status_labels = ['moderate'] * node_num
        batch_samples_info = [status_labels]
        batch_samples_tokens = [torch.tensor(self.tokenizer.encode(word)).cuda() for word in status_labels]

        return batch_samples_info, batch_samples_tokens
    
    

    def _status_promptor(self, predicted_labels, ground_labels):
        node_num = predicted_labels.size(1)
        BCE_crition = torch.nn.BCELoss(reduction='none')
        batch_samples_info = []
        batch_samples_tokens = []
        for node_idx in range(node_num):
            predict_node = predicted_labels[:, node_idx].unsqueeze(1)
            target_node = ground_labels[:, node_idx].unsqueeze(1)
            BCEloss_node = BCE_crition(predict_node, target_node).sum(1)
            loss_node_cls = BCEloss_node
            normalized_scores = 1 - torch.exp(-loss_node_cls * 1.0)
            status_words = ['exceptional ', 'proficient ', 'moderate', 'limited', 'inadequate']
            status_labels = [
                status_words[0] if score <= 0.1 else
                status_words[1] if score <= 0.2 else
                status_words[2] if score <= 0.8 else
                status_words[3] if score <= 0.9 else
                status_words[4] for score in normalized_scores
            ]
            batch_samples_info.append(status_labels)
        batch_samples_info = [list(row) for row in zip(*batch_samples_info)]
        for sample_status in batch_samples_info:
            batch_samples_tokens.append([torch.tensor(self.tokenizer.encode(word)).cuda() for word in sample_status])

        return batch_samples_info, batch_samples_tokens

    def _status_promptor_for_generation(self, predicted_labels):
        batch_size = predicted_labels.size(0)
        node_num = predicted_labels.size(1)
        # use 'moderate' for inference
        status_words = ['exceptional ', 'proficient ', 'moderate', 'limited', 'inadequate']
        status_labels = ['moderate'] * node_num
        batch_samples_tokens = []
        batch_samples_info = [status_labels] * batch_size
        for sample_status in batch_samples_info:
            batch_samples_tokens.append([torch.tensor(self.tokenizer.encode(word)).cuda() for word in sample_status])

        return batch_samples_info, batch_samples_tokens

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            image_features: torch.FloatTensor = None,
            image_id: torch.LongTensor = None,
            pathological_graph: torch.FloatTensor = None,
            pathological_label: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
    ):
        r"""
        image_features shape must be (num_images, num_image_patches, embed_dim)
        ```"""

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        pathology_cls_res = None

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)  # (b, text_len, 4096)
            batch_size = inputs_embeds.size(0)

            entity_list = self.Pathology_Balanced_Prompting_Module.pathology_entity_Chinese
            entity_embedding_list = []
            for entity_word in entity_list:
                entity_word_ids = torch.tensor(self.tokenizer.encode(entity_word)).cuda()
                entity_word_embeddings = self.get_input_embeddings()(entity_word_ids.unsqueeze(0).expand(batch_size, -1))  # (b, sub_token_size, 4096)
                entity_word_embeddings = entity_word_embeddings[:, 1:, :].mean(dim=1)  # (b, 4096)
                entity_embedding_list.append(entity_word_embeddings)
            entity_inputs = torch.stack(entity_embedding_list, dim=1)  # torch.Size([6, 33, 4096])

            visual_inputs = image_features  # torch.Size([6, 24, 2048])
            pathology_embeddings, pathology_cls_res = self.Pathology_Balanced_Prompting_Module(visual_inputs, entity_inputs, pathological_graph)  # 6, 33, 4096

            status_prompts_text, status_prompts_token = self._status_promptor_batch_avg(pathology_cls_res, pathological_label)
            status_embedding = []
            for each_entity_status in status_prompts_token:
                status_embedding.append(self.get_input_embeddings()(each_entity_status)[1:].mean(dim=0))
            status_embedding = torch.stack(status_embedding, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)  # (b, 32, 4096)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1:  # modified by zcx
                batch_id = image_id.tolist()
                visual_embeddings_att = []
                for local_sample_id in batch_id:
                    info_set, _, _ =  get_seetings()
                    att_path = info_set["visual_att_feature_dir_path"] + str(local_sample_id) + ".npz"
                    visual_embeddings_att.append(torch.from_numpy(np.load(att_path)['feat']))
                selected_image_feature = torch.stack(visual_embeddings_att).cuda()  # (b, 24, 14, 14, 2048)
                image_features = self.multi_modal_projector(selected_image_feature).reshape(-1, 1, 4096)  # (b*24, 1, 4096)

                # 2.1 Merge pathology embeddings
                pathology_embeddings = pathology_embeddings[:, 1:, :].reshape((-1, 1, 4096))
                image_features, pathology_embeddings = image_features.view(batch_size, 24, 4096), pathology_embeddings.view(batch_size, 32, 4096)
                embedding_prompts = torch.cat([image_features, pathology_embeddings], dim=1).reshape((-1, 1, 4096))
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_pathological_features(
                    embedding_prompts, inputs_embeds, input_ids, attention_mask, labels
                )  # pathology_embeddings(b*32, 1, 4096)  inputs_embeds(b, 255, 4096) input_ids(b, 255)  attention_mask(b, 255)  labels(b, 255)

                # status_embedding
                status_embedding = status_embedding.reshape((-1, 1, 4096))  # (b*32, 1, 4096)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_status_features(
                    status_embedding, inputs_embeds, input_ids, attention_mask, labels
                )  # pathology_embeddings(b*32, 1, 4096)  inputs_embeds(b, 255, 4096) input_ids(b, 255)  attention_mask(b, 255)  labels(b, 255)

                if labels is None:
                    labels = torch.full_like(attention_mask, self.ignore_index).to(torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            # elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,  # (b, text_len)
            position_ids=position_ids,  # (b, text_len)
            past_key_values=past_key_values,  # None
            inputs_embeds=inputs_embeds,  # (b, text_len, 4096)
            use_cache=use_cache,  # None
            output_attentions=output_attentions,  # False
            output_hidden_states=output_hidden_states,  # False
            return_dict=return_dict,  # True
        )

        logits = outputs[0]

        loss_cls = None
        if pathology_cls_res is not None:
            BCE_crition_objective = torch.nn.BCELoss(reduction='none')
            BCEloss_obj = BCE_crition_objective(pathology_cls_res, pathological_label).sum(1)
            loss_cls = BCEloss_obj.mean()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )
        print(f"Gen_loss:{loss}, CLS_loss:{loss_cls}")
        loss_lambda = 0.1
        loss = loss + loss_lambda * loss_cls
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids,
            past_key_values=None,
            inputs_embeds=None,
            # pixel_values=None,
            image_features=None,
            attention_mask=None,
            **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1:]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]):]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

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
                # "pixel_values": pixel_values, # origin
                "image_features": image_features,
            }
        )
        return model_inputs

    def generate(
            self,
            max_gen_len: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9,
            logprobs: bool = False,
            echo: bool = False,
            id: torch.IntTensor = None,  # just use to receive the sample id if have one
            input_ids: torch.LongTensor = None,
            # pixel_values: torch.FloatTensor = None,
            image_features: torch.FloatTensor = None,
            image_id: torch.LongTensor = None,
            pathological_graph: torch.FloatTensor = None,
            pathological_label: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            # vision_feature_layer: Optional[int] = None,
            # vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            input_ids (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            batch_size = inputs_embeds.size(0)

            entity_list = self.Pathology_Balanced_Prompting_Module.pathology_entity_Chinese
            entity_embedding_list = []
            for entity_word in entity_list:
                entity_word_ids = torch.tensor(self.tokenizer.encode(entity_word)).cuda()
                entity_word_embeddings = self.get_input_embeddings()(
                    entity_word_ids.unsqueeze(0).expand(batch_size, -1))  # (b, sub_token_size, 4096)
                entity_word_embeddings = entity_word_embeddings[:, 1:, :].mean(dim=1)  # (b, 4096)
                entity_embedding_list.append(entity_word_embeddings)
            entity_inputs = torch.stack(entity_embedding_list, dim=1)  # torch.Size([6, 33, 4096])

            visual_inputs = image_features  # torch.Size([6, 24, 2048])
            pathology_embeddings, pathology_cls_res = self.Pathology_Balanced_Prompting_Module(visual_inputs, entity_inputs, pathological_graph, image_id.tolist())  # 6, 33, 4096

            status_prompts_text, status_prompts_token = self._status_promptor_for_generation_batch_avg(pathology_cls_res) # 采用matched word作为state prompt
            status_embedding = []
            for each_entity_status in status_prompts_token:
                status_embedding.append(self.get_input_embeddings()(each_entity_status)[1:].mean(dim=0))
            status_embedding = torch.stack(status_embedding, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)  # (b, 32, 4096)

            # 2. Merge text and images
            # if pixel_values is not None and input_ids.shape[1] != 1: # origin
            if image_features is not None and input_ids.shape[1] != 1:  # modified by zcx
                batch_id = image_id.tolist()
                visual_embeddings_att = []
                for local_sample_id in batch_id:
                    info_set, _, _ =  get_seetings()
                    att_path = info_set["visual_att_feature_dir_path"] + str(local_sample_id) + ".npz"
                    visual_embeddings_att.append(torch.from_numpy(np.load(att_path)['feat']))
                selected_image_feature = torch.stack(visual_embeddings_att).cuda()  # (b, 24, 14, 14, 2048)
                image_features = self.multi_modal_projector(selected_image_feature).reshape(-1, 1, 4096)  # (b*24, 1, 4096)

                pathology_embeddings = pathology_embeddings[:, 1:, :].reshape((-1, 1, 4096))  # 丢弃全局节点 (b*32, 1, 4096)
                image_features, pathology_embeddings = image_features.view(batch_size, 24, 4096), pathology_embeddings.view(batch_size, 32, 4096)
                embedding_prompts = torch.cat([image_features, pathology_embeddings], dim=1).reshape((-1, 1, 4096))
                inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_pathological_features(
                    embedding_prompts, inputs_embeds, input_ids, attention_mask, None
                )  # pathology_embeddings(b*32, 1, 4096)  inputs_embeds(b, 255, 4096) input_ids(b, 255)  attention_mask(b, 255)  labels(b, 255)

                status_embedding = status_embedding.reshape((-1, 1, 4096))  # (b*32, 1, 4096)
                inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_status_features(
                    status_embedding, inputs_embeds, input_ids, attention_mask, None
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            # elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        bsz = len(inputs_embeds)
        min_prompt_len = min(len(t) for t in inputs_embeds)
        max_prompt_len = max(len(t) for t in inputs_embeds)
        total_len = max_gen_len + max_prompt_len

        pad_id = self.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.int, device="cuda")
        embeds = torch.full((bsz, total_len, inputs_embeds.shape[-1]), 0, dtype=torch.float, device="cuda")
        for k, t in enumerate(inputs_embeds):
            embeds[k, : len(t)] = t
        for k, t in enumerate(input_ids):
            tokens[k, : len(t)] = t

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        stop_tokens = torch.tensor([self.stop_token_id], device="cuda")

        for cur_pos in range(min_prompt_len, total_len):
            outputs = self.language_model(
                past_key_values=past_key_values,
                inputs_embeds=embeds[:, prev_pos:cur_pos],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            logits = outputs.logits  # 30 256 128265
            past_key_values = outputs.past_key_values

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            next_token_embed = self.get_input_embeddings()(next_token)
            embeds[:, cur_pos] = next_token_embed
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        # if logprobs:
        #     token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(input_ids[i])
            toks = toks[start: len(input_ids[i]) + max_gen_len]
            probs = None
            # if logprobs:
            #     probs = token_logprobs[i][start : len(input_ids[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in [self.stop_token_id]:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    # probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            # out_logprobs.append(probs)
        # return (out_tokens, out_logprobs if logprobs else None)
        return out_tokens

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        projector_params = {name: param for name, param in self.named_parameters() if 'multi_modal_projector' in name}
        pathology_module_names = {name: param for name, param in self.named_parameters() if 'Pathology_Balanced_Prompting_Module' in name}
        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        save_params = {**projector_params, **lora_params, **pathology_module_names}
        save_path = os.path.join(save_directory, self.save_peft_model_name)

        torch.save(save_params, save_path)
        print(f"Model weights saved to {save_path}")

    def from_pretrained(self, load_directory):
        load_params = torch.load(os.path.join(load_directory, self.save_peft_model_name))
        self.load_state_dict(load_params, strict=False)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token