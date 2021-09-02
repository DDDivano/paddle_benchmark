#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import paddle
import torch
import numpy as np
from douyar import Douyar


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


if __name__ == '__main__':
    test_abs()
    test_matmul()
