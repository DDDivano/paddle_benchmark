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
import time
from paddle import to_tensor


class Jelly(object):
    """
    compare tools
    """
    def __init__(self, paddle_api, torch_api):
        self.seed = 33
        self.enable_backward = True
        self.debug = True
        paddle.set_default_dtype(np.float64)
        torch.set_default_dtype(torch.float64)

        self.times = 10000
        self.paddle_forward_time = []
        self.paddle_backward_time = []
        self.torch_forward_time = []
        self.torch_backward_time = []

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
            start_forward = time.perf_counter()
            res = self.paddle_api(**self.paddle_param)
            end_forward = time.perf_counter()
            self.paddle_forward_time.append(end_forward - start_forward)
            return res
        elif self._layertypes(self.paddle_api) == "class":
            obj = self.paddle_api(**self.paddle_param)
            start_forward = time.perf_counter()
            res = obj(self.paddle_data)
            end_forward = time.perf_counter()
            self.paddle_forward_time.append(end_forward - start_forward)
            return res
        else:
            raise AttributeError

    def paddle_backward(self, res):
        loss = paddle.mean(res)
        start_backward = time.perf_counter()
        loss.backward()
        end_backward = time.perf_counter()
        self.paddle_backward_time.append(end_backward - start_backward)
        

    def run(self):
        if self.times < 10 and self.times % 10 != 0:
            raise Exception("run times must be a multiple of 10")
        logging.info("start compare [paddle]{} and [torch]{}".format(str(self.paddle_api.__name__), str(self.torch_api.__name__)))
        for place in self.places:
            for _ in range(self.times):
                paddle.set_device(place)
                self._run_paddle()
                if place == "cpu":
                    torch.device("cpu")
                else:
                    torch.device(0)
                self._run_torch()
            self._show()


    def _run_paddle(self):
        res = self.paddle_forward()
        # logging.info("[paddle_forward] is {}".format(res.numpy()))
        grad = self.paddle_backward(res)
        # logging.info("[paddle_grad] is {}".format(grad))
        

    def _run_torch(self):
        res = self.torch_forward()
        # logging.info("[torch_forward] is {}".format(res.detach().numpy()))
        grad = self.torch_backward(res)
        # logging.info("[torch_grad] is {}".format(grad))
        return res.detach().numpy(), grad

    def torch_forward(self):
        if self._layertypes(self.torch_api) == "func":
            start_forward = time.perf_counter()
            res = self.torch_api(**self.torch_param)
            end_forward = time.perf_counter()
            self.torch_forward_time.append(end_forward - start_forward)
            return res
        elif self._layertypes(self.torch_api) == "class":
            obj = self.torch_api(**self.torch_param)
            start_forward = time.perf_counter()
            res = obj(self.torch_data)
            end_forward = time.perf_counter()
            self.torch_forward_time.append(end_forward - start_forward)
            return res
        else:
            raise AttributeError

    def torch_backward(self, res):
        loss = torch.mean(res)
        start_backward = time.perf_counter()
        loss.backward()
        end_backward = time.perf_counter()
        self.torch_backward_time.append(end_backward - start_backward)
       

    def _show(self):
        # 去掉最高和最低的 1/10 剩下的比较
        head = int(self.times/10)
        tail = int(self.times - self.times/10)
        logging.info("paddle {} times forward cost {}".format(tail-head, sum(sorted(self.paddle_forward_time)[head:tail])))
        logging.info("paddle {} times backward cost {}".format(tail-head, sum(sorted(self.paddle_backward_time)[head:tail])))
        logging.info("torch {} times forward cost {}".format(tail-head, sum(sorted(self.torch_forward_time)[head:tail])))
        logging.info("torch {} times forward cost {}".format(tail-head, sum(sorted(self.torch_backward_time)[head:tail])))

        if sum(sorted(self.paddle_forward_time)[head:tail]) > sum(sorted(self.torch_forward_time)[head:tail]):
            forward = "torch forward is {:.3f}x faster than paddle".format(sum(sorted(self.paddle_forward_time)[head:tail]) / sum(sorted(self.torch_forward_time)[head:tail]))
        else:
            forward = "paddle forward is {:.3f}x faster than torch".format(sum(sorted(self.torch_forward_time)[head:tail]) / sum(sorted(self.paddle_forward_time)[head:tail]))
        logging.info(forward)

        if sum(sorted(self.paddle_backward_time)[head:tail]) > sum(sorted(self.torch_backward_time)[head:tail]):
            backward = "torch backward is {:.3f}x faster than paddle".format(sum(sorted(self.paddle_backward_time)[head:tail]) / sum(sorted(self.torch_backward_time)[head:tail]))
        else:
            backward = "paddle backward is {:.3f}x faster than torch".format(sum(sorted(self.torch_backward_time)[head:tail]) / sum(sorted(self.paddle_backward_time)[head:tail]))
        logging.info(backward)

        total_paddle = sum(sorted(self.paddle_backward_time)[head:tail]) + sum(sorted(self.paddle_forward_time)[head:tail])
        total_torch = sum(sorted(self.torch_backward_time)[head:tail]) + sum(sorted(self.torch_forward_time)[head:tail])

        if total_paddle > total_torch:
            total = "Total: torch is {:.3f}x faster than paddle".format(total_paddle / total_torch)
        else:
            total = "Total: paddle is {:.3f}x faster than torch".format(total_torch / total_paddle)
        logging.info(total)
        # print(sum(sorted(self.paddle_forward_time)[head:tail]))
        # print(sum(sorted(self.paddle_backward_time)[head:tail]))
        # print(sum(sorted(self.torch_forward_time)[head:tail]))
        # print(sum(sorted(self.torch_backward_time)[head:tail]))

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
            except Exception as e:
                logging.error("params are not same in paddle and torch. please set compare_dict to check grad result")
        else:
            pass


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