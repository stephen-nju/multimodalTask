# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: snipets.py
@time: 2022/9/29 19:22
"""
import six

is_py2 = six.PY2


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
