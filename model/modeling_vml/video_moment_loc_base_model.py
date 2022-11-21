# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: video_moment_loc_base_model.py
@time: 2022/11/1 15:43
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, video_feature, text_feature):
        return torch.cat([video_feature, text_feature], dim=-1)


class PositionEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, num_embeddings, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class AttentionFlow(nn.Module):
    """
    参考 Bidirectional Attention Flow for Machine Comprehension
    构建attention query and context attention
    由于这里的query向量简化成一维，所以context to query 直接拼接，
    query to context 计算下attention
    """

    def __init__(self,
                 hidden_size,
                 drop_rate,
                 ):
        super().__init__()
        self.drop_out = nn.Dropout(p=drop_rate)
        self.w = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size)
        self.query_to_context_attention = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, context, query):
        # context经过了lstm编码
        # context [batch_size,seq_length,hidden_size*2]
        # query [batch_size,hidden_size]
        seq_length = context.shape[1]
        query = torch.unsqueeze(query, dim=1).repeat(1, seq_length, 1)
        energy = self.w(torch.cat([context, query], dim=-1))
        energy = self.drop_out(energy)
        energy = torch.relu(energy)
        attention = self.query_to_context_attention(energy).squeeze(-1)
        score = torch.softmax(attention, dim=1)
        # 矩阵乘法 p * m * s乘以p * s * n-> p * m * n
        x = torch.bmm(torch.transpose(context, 1, 2), torch.unsqueeze(score, -1))
        x = torch.transpose(x, 1, 2).repeat(1, seq_length, 1)
        # x shape[batch_size,seq_length,hidden_size*2]
        # return [batch_size,seq_length,hidden_size *5]
        return torch.cat([context, x, query], dim=-1)

    #
    # class ContextAttention(nn.Module):
    #     """
    #     构建query于video之间的交互
    #     """
    #
    #     def __init__(self,
    #                  embedding_dim,
    #                  hidden_size,
    #                  max_position_embeddings,
    #                  num_heads,
    #                  drop_rate,
    #                  ):
    #         super(ContextAttention, self).__init__()
    #         self.drop_out = nn.Dropout(p=drop_rate)
    #         self.layer_norm_input = nn.LayerNorm(hidden_size)
    #         self.layer_norm_output = nn.LayerNorm(hidden_size)
    #         self.pos_embedding = PositionEmbedding(num_embeddings=max_position_embeddings, embedding_dim=hidden_size)
    #         self.multi_att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
    #         self.linear = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
    #         self._init_weights()
    #
    #     def _init_weights(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 trunc_normal_(m.weight, std=.02)
    #                 if isinstance(m, nn.Linear) and m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.LayerNorm):
    #                 nn.init.constant_(m.bias, 0)
    #                 nn.init.constant_(m.weight, 1.0)
    #
    #     def forward(self, x, hidden_state):
    #         x = self.linear(x)
    #         # 计算注意力
    #         seq_length = hidden_state.shape[1]
    #         x = torch.unsqueeze(x, dim=1)
    #         x = x.expand(-1, seq_length, -1)
    #         hidden_state = hidden_state + self.pos_embedding(hidden_state)
    #         x = self.layer_norm_input(x)
    #         hidden_state = self.layer_norm_input(hidden_state)
    #         x, y = self.multi_att(x, hidden_state, hidden_state)
    #         x = x + hidden_state  # 残差连接
    #         x = self.layer_norm_output(x)
    #         x = self.drop_out(x)
    #         return x


class SimpleVideoMomentLoc(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 drop_rate,
                 num_layers=2,
                 out_features=2,
                 ):
        super(SimpleVideoMomentLoc, self).__init__()
        self.drop_rate = drop_rate
        self.video_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                                  bidirectional=True,
                                  dropout=self.drop_rate, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=out_features)
        self.drop_out = nn.Dropout(p=drop_rate)
        self.feature_fusion = FeatureFusion()
        self.layer_norm_input = nn.LayerNorm(embedding_dim)
        self.attention_flow = AttentionFlow(hidden_size=hidden_size,
                                            drop_rate=drop_rate
                                            )
        self.layer_norm_attention_out = nn.LayerNorm(hidden_size * 5)
        self.layer_norm_linear_input = nn.LayerNorm(hidden_size * 2)
        self.multimodal_lstm = nn.LSTM(input_size=hidden_size * 5, hidden_size=hidden_size, batch_first=True,
                                       bidirectional=True,
                                       dropout=self.drop_rate,
                                       num_layers=1
                                       )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, feature_video, feature_text, feature_query):
        x = self.feature_fusion(feature_video, feature_text)
        x = self.layer_norm_input(x)
        x = self.drop_out(x)
        # video的多模态特征进行编码
        video, _ = self.video_lstm(x)
        # (batch_size,seq_length,D*hidden_output) D=2 if bidirectional
        # 构建query与video的attention
        x = self.attention_flow(video, feature_query)

        x = self.layer_norm_attention_out(x)
        x, _ = self.multimodal_lstm(x)
        x = x + video
        x = self.layer_norm_linear_input(x)
        x = self.linear(x)
        return x
