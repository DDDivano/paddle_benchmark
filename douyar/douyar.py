#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
import torch
import numpy as np
from inspect import isclass
import random
import pytest
import logging
from paddle import to_tensor


class Douyar(object):
    """
    compare tools
    """
    def __init__(self, paddle_api, torch_api):
        self.seed = 33
        self.enable_backward = True
        self.debug = True
        paddle.set_default_dtype(np.float64)
        torch.set_default_dtype(torch.float64)

        self.paddle_api = paddle_api
        self.torch_api = torch_api
        self.compare_dict = None
        self.paddle_param = dict()
        self.paddle_data = None
        self.torch_param = dict()
        self.torch_data = None
        self.places = None
        self._set_seed()
        self._set_place()
        # 日志等级
        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        else:
            logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    def _set_seed(self):
        """
        init seed
        :return:
        """
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        use_cuda = paddle.is_compiled_with_cuda()
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

    def _set_place(self):
        """
        init place
        :return:
        """
        if paddle.is_compiled_with_cuda() is True:
            if torch.cuda.is_available() is True:
                self.places = ["cpu", "gpu:0"]
            else:
                raise EnvironmentError
        else:
            # default
            self.places = ["cpu"]


    def _layertypes(self, func):
        """
        define layertypes
        """
        types = {0: 'func', 1: "class"}
        # 设置函数执行方式，函数式还是声明式.
        if isclass(func):
            return types[1]
        else:
            return types[0]

    def set_paddle_param(self, data: dict):
        if "data" in data.keys():
            self.paddle_data = to_tensor(data["data"])
            # enable compute gradient
            self.paddle_data.stop_gradient = False
            del(data["data"])
        for k, v in data.items():
            if isinstance(v, (np.generic, np.ndarray)):
                if "ignore_var" in data.keys() and k not in data["ignore_var"] and k != "ignore_var" \
                        or "ignore_var" not in data.keys():
                    self.paddle_param[k] = to_tensor(v)
                    # enable compute gradient
                    self.paddle_param[k].stop_gradient = False
            else:
                self.paddle_param[k] = v

    def set_torch_param(self, data: dict):
        if "data" in data.keys():
            self.torch_data = torch.tensor(data["data"])
            # enable compute gradient
            self.torch_data.requires_grad = True
            del (data["data"])
        for k, v in data.items():
            if isinstance(v, (np.generic, np.ndarray)):
                if "ignore_var" in data.keys() and k not in data["ignore_var"] and k != "ignore_var" \
                        or "ignore_var" not in data.keys():
                    self.torch_param[k] = torch.tensor(v)
                    # enable compute gradient
                    self.torch_param[k].requires_grad = True
            else:
                self.torch_param[k] = v

    def paddle_forward(self):
        if self._layertypes(self.paddle_api) == "func":
            res = self.paddle_api(**self.paddle_param)
            return res
        elif self._layertypes(self.paddle_api) == "class":
            obj = self.paddle_api(**self.paddle_param)
            res = obj(self.paddle_data)
            return res
        else:
            raise AttributeError

    def paddle_backward(self, res):
        loss = paddle.mean(res)
        loss.backward()
        grad = {}
        for k, v in self.paddle_param.items():
            # 判断是不是Variable类型
            if isinstance(v, paddle.Tensor):
                grad[k] = v.grad
        if self._layertypes(self.paddle_api) == "class":
            grad["data"] = self.paddle_data.grad
        # grad["res"] = res.gradient()
        return grad

    def run(self):
        for place in self.places:
            logging.info("[{}]start compare [paddle]{} and [torch]{}".format(place, str(self.paddle_api.__name__),
                                                                         str(self.torch_api.__name__)))

            paddle.set_device(place)
            paddle_res = self._run_paddle()
            if place == "cpu":
                torch.device("cpu")
            else:
                torch.device(0)
            torch_res = self._run_torch()
            result = [paddle_res, torch_res]
            self._check(result)

    def _run_paddle(self):
        res = self.paddle_forward()
        # logging.info("[paddle_forward] is {}".format(res.numpy()))
        grad = self.paddle_backward(res)
        # logging.info("[paddle_grad] is {}".format(grad))
        return res.numpy(), grad

    def _run_torch(self):
        res = self.torch_forward()
        # logging.info("[torch_forward] is {}".format(res.detach().numpy()))
        grad = self.torch_backward(res)
        # logging.info("[torch_grad] is {}".format(grad))
        return res.detach().numpy(), grad

    def torch_forward(self):
        if self._layertypes(self.torch_api) == "func":
            res = self.torch_api(**self.torch_param)
            return res
        elif self._layertypes(self.torch_api) == "class":
            obj = self.torch_api(**self.torch_param)
            res = obj(self.torch_data)
            return res
        else:
            raise AttributeError

    def torch_backward(self, res):
        loss = torch.mean(res)
        loss.backward()
        grad = {}
        for k, v in self.torch_param.items():
            # 判断是不是Variable类型
            if isinstance(v, torch.Tensor):
                grad[k] = v.grad
        if self._layertypes(self.torch_api) == "class":
            grad["data"] = self.torch_data.grad
        # grad["res"] = res.gradient()
        return grad

    def _check(self, result):
        paddle_res = result[0]
        torch_res = result[1]
        logging.info("[check forward]")
        compare(paddle_res[0], torch_res[0])
        logging.info("check forward ==================>>>>  ok.")
        if self.compare_dict is not None and self.enable_backward:
            logging.info("[check backward]")
            if self.paddle_data is not None and self.torch_data is not None:
                paddle_tmp = paddle_res[1]["data"].numpy()
                torch_tmp = torch_res[1]["data"].numpy()
                logging.info("check grad ({} <=====> {})".format("data", "data"))
                compare(paddle_tmp, torch_tmp)
                logging.info("check grad ({} <=====> {}) ==================>>>> ok.".format("data", "data"))
            for paddle_var, torch_var in self.compare_dict.items():

                # 获取对应的var grad
                paddle_tmp = paddle_res[1][paddle_var]
                torch_tmp = torch_res[1][torch_var]
                if not isinstance(paddle_tmp, np.ndarray):
                    paddle_tmp = paddle_tmp.numpy()
                if not isinstance(torch_tmp, np.ndarray):
                    torch_tmp = torch_tmp.numpy()
                logging.info("check grad ({} <=====> {})".format(paddle_var, torch_var))
                compare(paddle_tmp, torch_tmp)
                logging.info("check grad ({} <=====> {}) ==================>>>> ok.".format(paddle_var, torch_var))
        elif self.compare_dict is None and self.enable_backward:
            try:
                logging.info("[check backward]")
                if self.paddle_data is not None and self.torch_data is not None:
                    paddle_tmp = paddle_res[1]["data"].numpy()
                    torch_tmp = torch_res[1]["data"].numpy()
                    logging.info("check grad ({} <=====> {})".format("data", "data"))
                    compare(paddle_tmp, torch_tmp)
                    logging.info("check grad ({} <=====> {}) ==================>>>> ok.".format("data", "data"))
                for k, v in self.paddle_param.items():
                    # 判断是不是Variable类型
                    if isinstance(v, paddle.Tensor):
                        # 获取对应的var grad
                        paddle_tmp = paddle_res[1][k]
                        torch_tmp = torch_res[1][k]
                        if not isinstance(paddle_tmp, np.ndarray):
                            paddle_tmp = paddle_tmp.numpy()
                        if not isinstance(torch_tmp, np.ndarray):
                            torch_tmp = torch_tmp.numpy()
                        logging.info("check grad ({} <=====> {})".format(k, k))
                        compare(paddle_tmp, torch_tmp)
                        logging.info("check grad ({} <=====> {}) ==================>>>> ok.".format(k, k))
            except AssertionError as e:
                pass
            except Exception as e:
                print(e)

                logging.error("params are not same in paddle and torch. please set compare_dict to check grad result")
        else:
            pass


def compare(paddle, torch, delta=1e-6, rtol=0):
    """
    比较函数
    :param paddle: paddle结果
    :param torch: torch结果
    :param delta: 误差值
    :return:
    """
    if isinstance(paddle, np.ndarray):
        expect = np.array(torch)
        res = np.allclose(paddle, torch, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logging.error("the paddle is {}".format(paddle))
            logging.error("the torch is {}".format(torch))
        # tools.assert_true(res)
        assert res
        # tools.assert_equal(result.shape, expect.shape)
        assert paddle.shape == expect.shape
    elif isinstance(paddle, list):
        for i, j in enumerate(paddle):
            if isinstance(j, (np.generic, np.ndarray)):
                compare(j, torch[i], delta, rtol)
            else:
                compare(j.numpy(), torch[i], delta, rtol)
    elif isinstance(paddle, str):
        res = paddle == torch
        if res is False:
            logging.error("the paddle is {}".format(paddle))
            logging.error("the torch is {}".format(torch))
        assert res
    else:
        assert paddle == pytest.approx(torch, delta)
        # tools.assert_almost_equal(result, expect, delta=delta)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def get_dict(**data):
    return dict(data)