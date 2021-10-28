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




def linear32():
    """
    test linear
    :return:
    """
    x = randtool("float32", -10, 10, [630, 888])
    w = randtool("float32", -10, 10, [888, 633])
    b = randtool("float32", -10, 10, [633])
    obj = Douyar(paddle_api=paddle.nn.functional.linear, torch_api=torch.nn.functional.linear, default_type="float32")
    paddle_param = dict({"x": x, "weight": w, "bias": b})
    torch_param = dict({"input": x, "weight": w.transpose([1, 0]), "bias": b})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input"})
    obj.run()


def linear64():
    """
    test linear
    :return:
    """
    x = randtool("float64", -10, 10, [630, 888])
    w = randtool("float64", -10, 10, [888, 633])
    b = randtool("float64", -10, 10, [633])
    obj = Douyar(paddle_api=paddle.nn.functional.linear, torch_api=torch.nn.functional.linear, default_type="float64")
    paddle_param = dict({"x": x, "weight": w, "bias": b})
    torch_param = dict({"input": x, "weight": w.transpose([1, 0]), "bias": b})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.compare_dict = dict({"x": "input"})
    obj.run()


if __name__ == '__main__':
    linear32()
    linear64()