#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle
import torch
import numpy as np
from jelly import Jelly
from jelly import randtool
from jelly import get_dict


def matmul():
    obj = Jelly(paddle_api=paddle.matmul, torch_api=torch.matmul)
    data = randtool("float", 0, 1, [1])
    data1 = randtool("float", 0, 1, [1])
    paddle_param = get_dict(x=data, y=data1)
    torch_param = get_dict(input=data, other=data1)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.times = 100
    obj.run()


if __name__ == '__main__':
    matmul()
