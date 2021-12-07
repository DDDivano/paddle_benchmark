import paddle
import torch
import numpy as np
from jelly import Jelly
from jelly import randtool
from jelly import get_dict


def multiply():
    obj = Jelly(paddle_api=paddle.multiply, torch_api=torch.multiply)
    paddle_param = dict({"x": np.array([1.0]), "y": np.array([1.0])})
    torch_param = dict({"input": np.array([1.0]), "other": np.array([1.0])})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    multiply()
