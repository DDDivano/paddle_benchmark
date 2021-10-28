#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle
import torch
import numpy as np
from douyar import Douyar
from douyar import randtool
from douyar import get_dict




def layernorm():
    """
    test linear
    :return:
    """
    x = randtool("float", -10, 10, [207, 2, 1024])
    obj = Douyar(paddle_api=paddle.nn.LayerNorm, torch_api=torch.nn.LayerNorm)
    paddle_param = dict({"data": x, "normalized_shape": 1024})
    torch_param = dict({"data": x, "normalized_shape": 1024})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input"})
    obj.run()


def layernorm32():
    """
    test linear
    :return:
    """
    x = randtool("float32", -10, 10, [207, 2, 1024])
    obj = Douyar(paddle_api=paddle.nn.LayerNorm, torch_api=torch.nn.LayerNorm, default_type="float32")
    paddle_param = dict({"data": x, "normalized_shape": 1024})
    torch_param = dict({"data": x, "normalized_shape": 1024})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input"})
    obj.run()


def layernorm64():
    """
    test linear
    :return:
    """
    x = randtool("float64", -10, 10, [207, 2, 1024])
    obj = Douyar(paddle_api=paddle.nn.LayerNorm, torch_api=torch.nn.LayerNorm, default_type="float64")
    paddle_param = dict({"data": x, "normalized_shape": 1024})
    torch_param = dict({"data": x, "normalized_shape": 1024})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    # obj.compare_dict = dict({"x": "input"})
    obj.run()

if __name__ == '__main__':
    # layernorm32()
    layernorm64()