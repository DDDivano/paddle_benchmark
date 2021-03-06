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




def matmul32():
    """
    test linear
    :return:
    """
    x = randtool("float32", -10, 10, [10, 30, 4])
    y = randtool("float32", -10, 10, [4, 50])
    obj = Douyar(paddle_api=paddle.matmul, torch_api=torch.matmul, default_type="float32")
    paddle_param = dict({"x": x, "y": y})
    torch_param = dict({"input": x, "other": y})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()


def matmul64():
    """
    test linear
    :return:
    """
    x = randtool("float64", -10, 10, [10, 30, 4])
    y = randtool("float64", -10, 10, [4, 50])
    obj = Douyar(paddle_api=paddle.matmul, torch_api=torch.matmul, default_type="float64")
    paddle_param = dict({"x": x, "y": y})
    torch_param = dict({"input": x, "other": y})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input", "y": "other"})
    obj.run()


if __name__ == '__main__':
    matmul32()
    matmul64()