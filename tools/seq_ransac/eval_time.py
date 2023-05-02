import time
import torch
import colorama


class EvalTime(object):
    def __init__(self, disable=False):
        self.last_time = None
        self.disable = disable
        self.data = {}
        self.count = {}
        self.warm_up = {}
        self.warm_up_it = 3

    def __call__(self, info):
        if not self.disable:
            torch.cuda.synchronize()
            t = time.perf_counter()
            if self.last_time is None:
                self.last_time = t
                print("{}info : {}{} : %f".format(colorama.Fore.CYAN, info, colorama.Style.RESET_ALL) % t)
            else:
                if info not in self.data.keys():
                    self.data[info] = (t - self.last_time) * 1000
                    self.count[info] = 1
                    self.warm_up[info] = 1
                elif self.warm_up[info] < self.warm_up_it:
                    self.data[info] = (t - self.last_time) * 1000
                    self.count[info] = 1
                    self.warm_up[info] += 1
                else:
                    self.data[info] += (t - self.last_time) * 1000
                    self.count[info] += 1

                print(
                    "{}info : {}{}".format(colorama.Fore.CYAN, info,
                                           colorama.Style.RESET_ALL) + ' : % f, {}interval{}: % f'.format(
                        colorama.Fore.RED,
                        colorama.Style.RESET_ALL) % (
                        t, (t - self.last_time) * 1000) + ', {}average{}: % f'.format(colorama.Fore.RED,
                                                                                       colorama.Style.RESET_ALL) % (
                            self.data[info] / self.count[info]))
                self.last_time = t