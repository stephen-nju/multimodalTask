import argparse

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.modeling_drn.fcos import FCOSModule


def apply_mask1d(attention, image_locs):
    batch_size, num_loc = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = torch.ge(tmp1, tmp2)
    attention = attention.masked_fill(mask, -1e30)
    return attention


def conv_with_kaiming_uniform(use_bn=True, use_relu=True, use_dropout=False):
    def make_conv(
            in_channels, out_channels, kernel_size=3, stride=1, dilation=1
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_bn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        # nn.init.constant_(conv.bias, 0)
        module = [conv, ]
        if use_bn:
            module.append(nn.BatchNorm1d(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if use_dropout:
            module.append(nn.Dropout(p=0.5))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


class MultiLevelAttention(nn.Module):

    def __init__(self, hidden_size, num_level=3):
        super(MultiLevelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_level = num_level
        self.W1 = Linear(self.hidden_size * 2, 1)
        for i in range(num_level):
            W2 = Linear(self.hidden_size, self.hidden_size * 2)
            setattr(self, "W2_%d" % i, W2)
        self.W3 = Linear(self.hidden_size * 4, hidden_size * 2)

    def forward(self, hidden, encoder_outputs):
        """

        Arguments:
            hidden  -- shape=[1,batch_size,hidden_size*4]
            encoder_outputs shape=[batch_size,max_length,encode_dim]

        Returns:
            [type] -- [description]
        """
        # TODO check attention的计算
        output = []
        # hidden向量进行降维
        hidden = F.relu(self.W3(hidden))
        hidden = hidden.expand_as(encoder_outputs)
        # 计算attention
        for i in range(self.num_level):
            attention = getattr(self, "W2%d" % i)
            energy = attention(torch.cat((hidden, encoder_outputs), dim=-1))
            energy = self.W1(F.tanh(energy))
            attention_score = F.softmax(energy.squeeze(), dim=-1).unsqueeze(1)
            output_feature = torch.bmm(encoder_outputs, attention_score)
            output.append(output_feature)
        return output


class Backbone(nn.Module):
    def __init__(self, channels_list, conv_block):
        super(Backbone, self).__init__()

        self.num_layers = len(channels_list)
        self.blocks = []
        for idx, channels_config in enumerate(channels_list):
            block = "forward_conv{}".format(idx)
            block_module = conv_block(channels_config[0], channels_config[1],
                                      kernel_size=channels_config[2], stride=channels_config[3])
            self.add_module(block, block_module)
            self.blocks.append(block)

    def forward(self, x, query_fts, position_fts):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        results = []

        for idx in range(self.num_layers):
            query_ft = query_fts[idx].unsqueeze(1).permute(0, 2, 1)
            position_ft = position_fts[idx]
            x = query_ft * x
            if idx == 0:
                x = torch.cat([x, position_ft], dim=1)
            x = self._modules['forward_conv{}'.format(idx)](x)
            results.append(x)

        return tuple(results)


class QueryEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, embed_dim=300, num_layers=1, bidirection=True):
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.biLSTM = nn.LSTM(embed_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)
        self.build_extract_textual_command()

    def build_extract_textual_command(self):
        self.qInput = Linear(self.hidden_dim * 4, self.hidden_dim)
        for t in range(3):
            qInput_layer2 = Linear(self.hidden_dim, self.hidden_dim * 2)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = Linear(self.hidden_dim * 2, 1)

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        # act_fun = ops.activations['RELU']
        q_cmd = qInput_layer2(F.relu(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def forward(self, query_tokens, query_length):

        outputs = []
        query_embedding = self.embedding(query_tokens)
        query_embedding = pack_padded_sequence(query_embedding, query_length, batch_first=True)
        self.biLSTM.flatten_parameters()
        # TODO: h_0, c_0 is zero here
        output, _ = self.biLSTM(query_embedding)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # select the hidden state of the last word individually, since the lengths of query are variable.
        q_vector_list = []
        batch_size = query_length.size(0)
        for i, length in enumerate(query_length):
            h1 = output[i][0]
            hs = output[i][length - 1]
            q_vector = torch.cat((h1, hs), dim=-1)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)

        for cmd_t in range(3):
            outputs.append(self.extract_textual_command(q_vector, output, query_length, cmd_t))
        # output = self.textualAttention(output, q_vector, query_length)

        # Note: the output here is zero-padded, we need slice the non-zero items for the following operations.
        return outputs


# class QueryEncoder(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, embed_dim, lstm_layers, bi_direction):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.hidden_dim = hidden_dim
#         self.embed_dim = embed_dim
#         self.bi_direction = bi_direction
#         self.token_embedding = nn.Embedding(self.vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=self.hidden_dim, bidirectional=True,
#                             num_layers=lstm_layers, dropout=0.0, batch_first=True)
#         self.multilevel_attention = MultiLevelAttention(hidden_size=hidden_dim, num_level=3)
#
#     def forward(self, query_tokens, query_length):
#         # query_tokens=[batch_size,length]
#         # query_length=[batch_size]
#         outputs = []
#         query_embedding = self.token_embedding(query_tokens)
#         query_embedding = pack_padded_sequence(query_embedding, query_length, batch_first=True)
#         self.lstm.flatten_parameters()
#         output, _ = self.lstm(query_embedding)
#         output, _ = pad_packed_sequence(output, batch_first=True)
#         # select the hidden state of the last word individually, since the lengths of query are variable.
#         q_vector_list = []
#         for i, length in enumerate(query_length):
#             h1 = output[i][0]
#             hs = output[i][length - 1]
#             q_vector = torch.cat((h1, hs), dim=-1)
#             q_vector_list.append(q_vector)
#         q_vector = torch.stack(q_vector_list)
#         outputs = self.multilevel_attention(hidden=q_vector, encoder_outputs=output)
#         # Note: the output here is zero-padded, we need slice the non-zero items for the following operations.
#         return outputs


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        # add name of module into lists, inner: 1x1 conv, layer: 3x3 conv
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            # add module named as *_block into self-module
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # process the last lowest resolution feat and first feed it into 1 x 1 conv
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        # exclude the last one and process the feat from the second highest layer feat
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv1d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv1d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


# DRN 构建
class DenseRegressionNet(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size):
        """

        :rtype: object
        """
        super(DenseRegressionNet, self).__init__()
        self.first_output_dim = args.first_output_dim
        self.fpn_feature_dim = args.fpn_feature_dim
        self.feature_dim = args.feature_dim
        self.query_encoder = QueryEncoder(vocab_size, args.hidden_dim, args.embed_dim, args.lstm_layers,
                                          bidirection=True)

        channels_list = [
            (self.feature_dim + 256, self.first_output_dim, 3, 1),
            (self.first_output_dim, self.first_output_dim * 2, 3, 2),
            ((self.first_output_dim * 2), self.first_output_dim * 4, 3, 2),
        ]
        conv_func = conv_with_kaiming_uniform(use_bn=True, use_relu=True)
        self.backbone_net = Backbone(channels_list, conv_func)
        self.fpn = FPN([256, 512, 1024], 512, conv_func)
        self.fcos = FCOSModule(args, self.fpn_feature_dim)
        self.prop_fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.position_transform = nn.Linear(3, 256)

        for t in range(len(channels_list)):
            if t > 0:
                setattr(self, "qInput%d" % t, nn.Linear(1024, channels_list[t - 1][1]))
            else:
                setattr(self, "qInput%d" % t, nn.Linear(1024, self.feature_dim))

    def forward(self, query_tokens, query_length, props_features,
                props_start_end, gt_start_end, props_num, num_frames):

        position_info = [props_start_end, props_start_end]
        position_feats = []
        query_features = self.query_encoder(query_tokens, query_length)
        for i in range(len(query_features)):
            query_fc = getattr(self, "qInput%d" % i)
            query_features[i] = query_fc(query_features[i])
            if i > 1:
                position_info.append(
                    torch.cat([props_start_end[:, :: 2 * (i - 1), [0]], props_start_end[:, 1:: 2 * (i - 1), [1]]],
                              dim=-1))
            props_duration = (position_info[i][:, :, 1] - position_info[i][:, :, 0]).unsqueeze(-1)
            position_feat = torch.cat((position_info[i], props_duration), dim=-1).float()
            position_feats.append(self.position_transform(position_feat).permute(0, 2, 1))

        props_features = self.prop_fc(props_features)
        inputs = props_features.permute(0, 2, 1)
        outputs = self.backbone_net(inputs, query_features, position_feats)
        outputs = self.fpn(outputs)

        box_lists, loss_dict = self.fcos(outputs, gt_start_end.float())

        return box_lists, loss_dict
