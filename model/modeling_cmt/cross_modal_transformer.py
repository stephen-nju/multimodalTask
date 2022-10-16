# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: cross_modal_transformer.py
@time: 2022/9/29 11:13
特征融合方式参考 https://github.com/zhanxlin/Product1M/tree/main/CAPTURE
相关文章  Product1M：Towards Weakly Supervised Instance-Level Product Retrieval via Cross-Modal Pretraining
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from collections import OrderedDict
from torch.nn import LayerNorm


# TODO
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
# layer norm epsilon 1e-12
#
#
#


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat(self.n_head, 1, 1)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 max_position_embeddings,
                 hidden_size,
                 hidden_dropout_prob
                 ):
        super(CrossEmbeddings, self).__init__(
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):
        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings  # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossPooler(nn.Module):
    def __init__(self, hidden_size):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossTransformerModel(nn.Module):
    def __init__(self,
                 max_position_embeddings,
                 hidden_dropout_prob,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads):
        super(CrossTransformerModel, self).__init__()

        self.embeddings = CrossEmbeddings(max_position_embeddings,
                                          hidden_size,
                                          hidden_dropout_prob)

        transformer_width = hidden_size
        transformer_layers = num_hidden_layers
        transformer_heads = num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, )
        self.pooler = CrossPooler(hidden_size)
        # self.apply(self.init_weights)

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def forward(self, concat_input, concat_type=None, attention_mask=None):

        # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch-tensor-to
        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1)).to(concat_input)
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask).to(concat_input)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output
