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


def conv2d():
    data = randtool("float", 0, 1, [1, 1, 1, 1])
    obj = Jelly(paddle_api=paddle.nn.functional.conv2d, torch_api=torch.nn.functional.conv2d)
    paddle_param = get_dict(x=data,
                            weight=np.ones(shape=[1, 1, 1, 1]).astype("float64") * 0.3,
                            bias=np.zeros(shape=[1]).astype("float64"),
                            stride=1,
                            padding=0,
                            )
    torch_param = get_dict(x=data,
                           weight=np.ones(shape=[1, 1, 1, 1]).astype("float64") * 0.3,
                           bias=np.zeros(shape=[1]).astype("float64"),
                           stride=1,
                           padding=0, )
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    conv2d()
