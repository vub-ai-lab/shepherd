import torch.multiprocessing as multiprocessing
import torch
import time
import os
import psutil
import gc

# Ensure that sharing all the models with all the workers is not too many FDs
multiprocessing.set_sharing_strategy('file_system')

def worker(to_worker, core, all_models):
    psutil.Process().cpu_affinity([core])
    torch.random.seed()

    # Execute functions
    while True:
        f, args = to_worker.get()

        for i in [0, 3]:
            args[i] = all_models[args[i]]

        f(args)

class Pool(object):
    def __init__(self, cores, maxsize, all_models):
        """ Initialize a pool with cores processes
        """
        self._to_worker = multiprocessing.Queue(maxsize)
        self._processes = []
        self._all_models = all_models

        affinity = list(psutil.Process().cpu_affinity())

        for i in range(cores):
            core = affinity[i % len(affinity)]
            p = multiprocessing.Process(target=worker, args=(self._to_worker, core, all_models), daemon=True)
            p.start()

            self._processes.append(p)

    def map(self, func, args):
        count = len(self._processes)

        # Push functions and arguments to the worker processes
        gc.disable()

        for arg in args:
            # Replace models by their index
            for i in [0, 3]:
                arg[i] = self._all_models.index(arg[i])

            self._to_worker.put((func, arg))

        gc.enable()

        return []
