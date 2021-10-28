#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import paddle
import torch
import numpy as np
from douyar import Douyar
from douyar import randtool
from douyar import get_dict

def conv2d32():
    data = randtool("float32", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0

    obj = Douyar(paddle_api=paddle.nn.functional.conv2d, torch_api=torch.nn.functional.conv2d, default_type="float32")
    paddle_param = get_dict(x = data,
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32") * 0.3,
    bias = np.zeros(shape = [1]).astype("float32"),
    stride = 1,
    padding = 0,
                            )
    torch_param = get_dict(x = data,
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32") * 0.3,
    bias = np.zeros(shape = [1]).astype("float32"),
    stride = 1,
    padding = 0,)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()


def conv2d64():
    data = randtool("float64", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0

    obj = Douyar(paddle_api=paddle.nn.functional.conv2d, torch_api=torch.nn.functional.conv2d, default_type="float64")
    paddle_param = get_dict(x = data,
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float64") * 0.3,
    bias = np.zeros(shape = [1]).astype("float64"),
    stride = 1,
    padding = 0,
                            )
    torch_param = get_dict(x = data,
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float64") * 0.3,
    bias = np.zeros(shape = [1]).astype("float64"),
    stride = 1,
    padding = 0,)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()

if __name__ == '__main__':
    conv2d32()
    conv2d64()