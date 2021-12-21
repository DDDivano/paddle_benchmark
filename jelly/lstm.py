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


def lstm():
    obj = Jelly(paddle_api=paddle.nn.LSTM, torch_api=torch.nn.LSTM)
    data = randtool("float", 0, 1, [1, 1, 1])
    paddle_param = get_dict(data=data, input_size=1, hidden_size=1, num_layers=1)
    torch_param = get_dict(data=data, input_size=1, hidden_size=1, num_layers=1)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    lstm()
