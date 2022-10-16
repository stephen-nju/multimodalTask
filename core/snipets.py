# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: snipets.py
@time: 2022/9/29 19:22
"""
import collections
import re

import six
import numpy as np
import torch

is_py2 = six.PY2

np_str_obj_array_pattern = re.compile(r'[SaUO]')
string_classes = (str, bytes)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_array_to_tensor(data):
    """
    tensor 或者np array 转化为tensor
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: convert_array_to_tensor(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: convert_array_to_tensor(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_array_to_tensor(d) for d in data))
    elif isinstance(data, tuple):
        return [convert_array_to_tensor(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([convert_array_to_tensor(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [convert_array_to_tensor(d) for d in data]
    else:
        return data


if __name__ == '__main__':
    a = [np.array([1, 2]), 3]
    v = convert_to_tensor(a)
    print(v)
