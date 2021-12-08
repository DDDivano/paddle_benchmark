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


def adaptivemaxpool1d():
    obj = Jelly(paddle_api=paddle.nn.AdaptiveMaxPool1D, torch_api=torch.nn.AdaptiveMaxPool1d)
    data = randtool("float", 0, 1, [1, 1, 2])
    paddle_param = get_dict(data=data, output_size=1)
    torch_param = get_dict(data=data, output_size=1)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    adaptivemaxpool1d()