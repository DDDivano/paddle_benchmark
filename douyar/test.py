#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import paddle
import torch
import numpy as np
from douyar import Douyar
from douyar import randtool
from douyar import get_dict


def test_abs():
    obj = Douyar(paddle_api=paddle.abs, torch_api=torch.abs)
    paddle_param = dict({"x": np.array([1.0, 2.0, 3.0])})
    torch_param = dict({"input": np.array([1.0, 2.0, 3.0])})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input"})
    obj.run()



def test_matmul():
    obj = Douyar(paddle_api=paddle.matmul, torch_api=torch.matmul)
    paddle_param = dict({"x": np.array([2.0, 2.0]), "y": np.array([[4.0], [4.0]])})
    torch_param = dict({"input": np.array([2.0, 2.0]), "other": np.array([[4.0], [4.0]])})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()


def test_conv2d():
    data = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0

    obj = Douyar(paddle_api=paddle.nn.functional.conv2d, torch_api=torch.nn.functional.conv2d)
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

def test_hardtanh():
    data = randtool("float", 0, 1, [2, 3, 4, 4])
    obj = Douyar(paddle_api=paddle.nn.Hardtanh, torch_api=torch.nn.Hardtanh)
    paddle_param = get_dict(data=data, min=- 1.0, max=1.0)
    torch_param = get_dict(data=data, min_val=-1.0, max_val=1.0,)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()

if __name__ == '__main__':
    test_abs()
    test_matmul()
    test_conv2d()
    test_hardtanh()