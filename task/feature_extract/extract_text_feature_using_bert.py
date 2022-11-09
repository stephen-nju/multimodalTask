# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: extract_text_feature_using_bert.py
@time: 2022/11/2 19:27
"""
import argparse
import json
import os
import shutil
import numpy as np
import math
from collections import defaultdict
import torch
from tqdm import tqdm

from transformers import BertTokenizer, BertConfig, BertModel


def loading_bert(args):
    bert_config = BertConfig.from_pretrained(args.bert_model)
    bert = BertModel.from_pretrained(args.bert_model,
                                     config=bert_config)
    return bert


if __name__ == '__main__':
    max_seq_length = 510
    parser = argparse.ArgumentParser(description="text feature extract")
    parser.add_argument("--bert_model", type=str, help="bert model path")
    parser.set_defaults(
        bert_model="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model"
    )
    args = parser.parse_args()
    model = loading_bert(args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    text_root = "/home/nlpbigdata/net_disk_project/zhubin/mmkg/text"
    text_feature_root = "/home/nlpbigdata/local_disk/text_feature"
    for text_name in tqdm(os.listdir(text_root)):
        video_name = text_name.strip(".txt")
        feature_dir = os.path.join(text_feature_root, video_name)
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)
        else:
            shutil.rmtree(feature_dir)
            os.mkdir(feature_dir)
        text_path = os.path.join(text_root, text_name)
        if os.path.isfile(text_path):
            with open(text_path, "r", encoding="utf-8") as g:
                data = json.load(g)
                word_piece = data["nbest"][0]["word_pieces"]
                # 构建index to time的字典
                time_to_index = defaultdict()
                # index_to_time = defaultdict()
                # # index不会重复
                for index, word in enumerate(word_piece):
                    time_value = math.ceil(word["start"] / 1000)
                    time_to_index.setdefault(time_value, []).append(index)
                word_length = len(word_piece)
                num_sentence_piece = math.ceil(word_length / max_seq_length)
                assert num_sentence_piece > 0
                if num_sentence_piece > 1:
                    feature_list = []
                    for index in range(num_sentence_piece):
                        if index == num_sentence_piece - 1:
                            sentence = word_piece[index * max_seq_length:]
                        else:
                            sentence = word_piece[index * max_seq_length:(index + 1) * max_seq_length]
                        # s为字典类型，sentence为列表类型
                        text = " ".join([s["word"] for s in sentence])
                        inputs = tokenizer(text, return_tensors="pt")
                        output = model(**inputs)
                        feat = output.last_hidden_state
                        # feat shape [batch_size,sequence,hidden_size]
                        # 移除cls 和sep
                        feat = feat[:, 1:-1, :]
                        feature_list.append(feat)
                    feat = torch.cat(feature_list, dim=1)
                else:
                    text = " ".join([s["word"] for s in word_piece])
                    inputs = tokenizer(text, return_tensors="pt")
                    output = model(**inputs)
                    feat = output.last_hidden_state
                feat = feat.detach().cpu().numpy()
                with open(os.path.join(feature_dir, "time_index.json"), "w", encoding="utf-8") as g:
                    json.dump(time_to_index, g, ensure_ascii=False)
                np.save(os.path.join(feature_dir, f"sentence_feature.npy"), feat)
