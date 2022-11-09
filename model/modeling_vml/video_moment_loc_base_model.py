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


class ContextAttention(nn.Module):
    """
    构建query于video之间的交互
    """

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 num_heads,
                 ):
        super(ContextAttention, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, hidden_state):
        x = self.linear(x)
        # 计算注意力
        seq_length = hidden_state.shape[1]
        x = torch.unsqueeze(x, dim=1)
        x = x.expand(-1, seq_length, -1)
        x, y = self.multi_att(x, hidden_state, hidden_state)
        return x


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
        self.bi_gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True,
                             dropout=self.drop_rate, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=out_features)
        self.drop_out = nn.Dropout(p=drop_rate)
        self.feature_fusion = FeatureFusion()
        self.context_attention = ContextAttention(embedding_dim=hidden_size, hidden_size=hidden_size * 2,
                                                  num_heads=4)
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

        x = self.drop_out(x)
        # video的序列特征
        x, _ = self.bi_gru(x)
        # (batch_size,seq_length,D*hidden_output) D=2 if bidirectional
        # 构建query与video的attention
        x = self.context_attention(feature_query, x)
        x = self.linear(x)
        return x
