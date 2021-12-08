#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
import torch
import json
import numpy as np
from inspect import isclass
import random
import logging
import time
import os
from paddle import to_tensor


class Jelly(object):
    """
    compare tools
    """
    def __init__(self, paddle_api, torch_api, place=None, card=None):
        self.seed = 33
        self.enable_backward = True
        self.debug = True
        paddle.set_default_dtype(np.float64)
        torch.set_default_dtype(torch.float64)

        self.times = 50000
        self.paddle_forward_time = []
        self.paddle_backward_time = []
        self.torch_forward_time = []
        self.torch_backward_time = []
        self.paddle_total_time = []
        self.torch_total_time = []

        self.dump_data = []
        self.result = {"paddle": dict(), "torch": dict()}

        self.paddle_api = paddle_api
        self.torch_api = torch_api
        self.compare_dict = None
        self.paddle_param = dict()
        self.paddle_data = None
        self.torch_param = dict()
        self.torch_data = None
        self.places = place
        self.card = card
        self._set_seed()
        self._set_place(self.card)
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

    def _set_place(self, card=None):
        """
        init place
        :return:
        """
        if self.places is None:
            if paddle.is_compiled_with_cuda() is True:
                if torch.cuda.is_available() is True:
                    if card is None:
                        paddle.set_device("gpu:0")
                        torch.device(0)
                    else:
                        paddle.set_device("gpu:{}".format(card))
                        torch.device(card)
                else:
                    raise EnvironmentError
            else:
                paddle.set_device("cpu")
                torch.device("cpu")
        else:
            if self.places == "cpu":
                paddle.set_device("cpu")
                torch.device("cpu")
            else:
                if card is None:
                    paddle.set_device("gpu:0")
                    torch.device(0)
                else:
                    paddle.set_device("gpu:{}".format(card))
                    torch.device(card)


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
        for _ in range(self.times):
            self._run_paddle()
        for _ in range(self.times):
            self._run_torch()
        self._compute()
        self._show()

    def _run_paddle(self):
        start_forward = time.perf_counter()
        res = self.paddle_forward()
        # logging.info("[paddle_forward] is {}".format(res.numpy()))
        grad = self.paddle_backward(res)
        # logging.info("[paddle_grad] is {}".format(grad))
        end_backward = time.perf_counter()
        self.paddle_total_time.append(end_backward - start_forward)
        
    def _run_torch(self):
        start_forward = time.perf_counter()
        res = self.torch_forward()
        # logging.info("[torch_forward] is {}".format(res.detach().numpy()))
        grad = self.torch_backward(res)
        # logging.info("[torch_grad] is {}".format(grad))
        end_backward = time.perf_counter()
        self.torch_total_time.append(end_backward - start_forward)


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

    def _compute(self):
        head = int(self.times / 20)
        tail = int(self.times - self.times / 20)
        self.result["paddle"]["forward"] = sum(sorted(self.paddle_forward_time)[head:tail])
        self.result["paddle"]["backward"] = sum(sorted(self.paddle_backward_time)[head:tail])
        self.result["paddle"]["total"] = sum(sorted(self.paddle_total_time)[head:tail])
        self.result["paddle"]["total_fb"] = self.result["paddle"]["forward"] + self.result["paddle"]["backward"]

        self.result["torch"]["forward"] = sum(sorted(self.torch_forward_time)[head:tail])
        self.result["torch"]["backward"] = sum(sorted(self.torch_backward_time)[head:tail])
        self.result["torch"]["total"] = sum(sorted(self.torch_total_time)[head:tail])
        self.result["torch"]["total_fb"] = self.result["torch"]["forward"] + self.result["torch"]["backward"]
        self._save(self.result)

    def _show(self):
        # 去掉最高和最低的 1/10 剩下的比较
        head = int(self.times/20)
        tail = int(self.times - self.times/20)
        logging.info("paddle {} times forward cost {:.5f}".format(tail-head, self.result["paddle"]["forward"]))
        logging.info("paddle {} times backward cost {:.5f}".format(tail-head, self.result["paddle"]["backward"]))
        logging.info("paddle {} times total cost {:.5f}".format(tail-head, self.result["paddle"]["total"]))
        logging.info("torch {} times forward cost {:.5f}".format(tail-head, self.result["torch"]["forward"]))
        logging.info("torch {} times backward cost {:.5f}".format(tail-head, self.result["torch"]["backward"]))
        logging.info("torch {} times total cost {:.5f}".format(tail-head, self.result["torch"]["total"]))

        if sum(sorted(self.paddle_forward_time)[head:tail]) > sum(sorted(self.torch_forward_time)[head:tail]):
            forward = "torch forward is {:.3f}x faster than paddle".format(self.result["paddle"]["forward"] / self.result["torch"]["forward"])
        else:
            forward = "paddle forward is {:.3f}x faster than torch".format(self.result["torch"]["forward"] / self.result["paddle"]["forward"])
        logging.info(forward)

        if sum(sorted(self.paddle_backward_time)[head:tail]) > sum(sorted(self.torch_backward_time)[head:tail]):
            backward = "torch backward is {:.3f}x faster than paddle".format(self.result["paddle"]["backward"] / self.result["torch"]["backward"])
        else:
            backward = "paddle backward is {:.3f}x faster than torch".format(self.result["torch"]["backward"] / self.result["paddle"]["backward"])
        logging.info(backward)

        if self.result["paddle"]["total_fb"] > self.result["torch"]["total_fb"]:
            total_fb = "Total_F+B: torch is {:.3f}x faster than paddle".format(self.result["paddle"]["total_fb"] / self.result["torch"]["total_fb"])
        else:
            total_fb = "Total_F+B: paddle is {:.3f}x faster than torch".format(self.result["torch"]["total_fb"] / self.result["paddle"]["total_fb"])
        logging.info(total_fb)

        if self.result["paddle"]["total"] > self.result["torch"]["total"]:
            total = "Total: torch is {:.3f}x faster than paddle".format(self.result["paddle"]["total"] / self.result["torch"]["total"])
        else:
            total = "Total: paddle is {:.3f}x faster than torch".format(self.result["torch"]["total"] / self.result["paddle"]["total"])
        logging.info(total)
        # print(sum(sorted(self.paddle_forward_time)[head:tail]))
        # print(sum(sorted(self.paddle_backward_time)[head:tail]))
        # print(sum(sorted(self.torch_forward_time)[head:tail]))
        # print(sum(sorted(self.torch_backward_time)[head:tail]))

    def _save(self, data):
        """
        保存数据到磁盘
        :return:
        """
        log_file = "./log/{}.json".format(str(self.paddle_api.__name__))
        if not os.path.exists("./log"):
            os.makedirs("./log")
        try:
            with open(log_file, 'w') as json_file:
                json.dump(data, json_file)
            logging.info("log save success!")
        except Exception as e:
            print(e)

    def dump(self):
        # 去掉最高和最低的 1/10 剩下的比较
        head = int(self.times/100)
        tail = int(self.times - self.times/100)
        paddle_forward = sum(sorted(self.paddle_forward_time)[head:tail])
        paddle_backward = sum(sorted(self.paddle_backward_time)[head:tail])
        torch_forward = sum(sorted(self.torch_forward_time)[head:tail])
        torch_backward = sum(sorted(self.torch_backward_time)[head:tail])

        if sum(sorted(self.paddle_forward_time)[head:tail]) > sum(sorted(self.torch_forward_time)[head:tail]):
            forward = "torch forward is {:.3f}x faster than paddle".format(sum(sorted(self.paddle_forward_time)[head:tail]) / sum(sorted(self.torch_forward_time)[head:tail]))
        else:
            forward = "paddle forward is {:.3f}x faster than torch".format(sum(sorted(self.torch_forward_time)[head:tail]) / sum(sorted(self.paddle_forward_time)[head:tail]))
        
        if sum(sorted(self.paddle_backward_time)[head:tail]) > sum(sorted(self.torch_backward_time)[head:tail]):
            backward = "torch backward is {:.3f}x faster than paddle".format(sum(sorted(self.paddle_backward_time)[head:tail]) / sum(sorted(self.torch_backward_time)[head:tail]))
        else:
            backward = "paddle backward is {:.3f}x faster than torch".format(sum(sorted(self.torch_backward_time)[head:tail]) / sum(sorted(self.paddle_backward_time)[head:tail]))

        total_paddle = sum(sorted(self.paddle_backward_time)[head:tail]) + sum(sorted(self.paddle_forward_time)[head:tail])
        total_torch = sum(sorted(self.torch_backward_time)[head:tail]) + sum(sorted(self.torch_forward_time)[head:tail])

        if total_paddle > total_torch:
            total = "Total: torch is {:.3f}x faster than paddle".format(total_paddle / total_torch)
        else:
            total = "Total: paddle is {:.3f}x faster than torch".format(total_torch / total_paddle)

        self.dump_data = [str(self.paddle_api.__name__), paddle_forward, paddle_backward, torch_forward, torch_backward, forward, backward, total]
        return self.dump_data


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