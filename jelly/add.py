import paddle
import torch
import numpy as np
from jelly import Jelly
from jelly import randtool
from jelly import get_dict


def add():
    obj = Jelly(paddle_api=paddle.add, torch_api=torch.add)
    paddle_param = dict({"x": np.array([1.0]), "y": np.array([1.0])})
    torch_param = dict({"input": np.array([1.0]), "other": np.array([1.0])})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


def add2():
    """
    大数据试验
    :return:
    """
    x = np.random.random(size=[50, 50, 50, 50])
    y = np.random.random(size=[50, 50, 50, 50])
    obj = Jelly(paddle_api=paddle.add, torch_api=torch.add)
    obj.times = 200
    paddle_param = dict({"x": x, "y": y})
    torch_param = dict({"input": x, "other": y})
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    add()
