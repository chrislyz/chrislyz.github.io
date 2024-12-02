---
layout: post
---

## Terminologies

A list of terminologies commonly used in distributed computing are explained in the following:

- *Device*[^1]: Primarily refers to GPU but can be any high-performance computation (HPC) accelerator.
- *Node*: A physical end (or a machine) or a container, consisting multiple devices.
- *Worker*: A worker in the context of distributed training.
- *Worker Group*: A set of workers that executes the same function.
- *Local Worker Group*: A subset of workers in the Worker Group running on the same node.
- *World Size*: The number of processes participating in the job, in which each process occupies a GPU. Hence, The total number of processes is equivalent to the number of GPUs, `world_size = torch.cuda.device_count()`.
  - $\textrm{world\_size} = \textrm{n\_nodes} \times \textrm{ngpu\_per\_node}$
- *Rank*: A unique identifier assigned to each process within a distributed process group. They are always consecutive integers ranging from 0 to `world_size-1`.
- *Local Rank*: The rank (id) of the process on the local machine.

[^1]: We mainly refer Device to GPUs in this note as the code demonstrated in the following are based on such assumption.

### Example

```
# e.g., supposing we have 2 nodes and 4 gpus each, 8 in total
# world size, local world size, node, rank, and local rank are as follows
# --------------------------------------------------
# world_size: |        8         |         8       |
# local_size: |        4         |         4       |
# node:       |     node 0       |      node 1     |
# rank:       | [0],[1],[2],[3]  | [4],[5],[6],[7] |
# local_rank: | [0],[1],[2],[3]  | [0],[1],[2],[3] |
# --------------------------------------------------
```

## Overview

PyTorch provides developers with 6 ways of training paradigms:

1. `torch.nn.DataParallel`: Single-device, if data and model can fit in **one** GPU,
2. `torch.nn.DataParallel`: Single-machine multi-GPU to make use of multiple GPUs to speed up training with minimal change of codes,
3. `torch.nn.parallel.DistributedDataParallel`: Single-machine multi-GPU to further speed up training with little change of codes,
4. `torch.nn.parallel.DistributedDataParallel`: Multi-machine launching script to scale up across machines,
5. `torch.distributed.FullyShardedDataParallel`: Multi-GPU training on a single-machine or multi-machine, if the data and model can not fit in one GPU,
6. `torch.distributed.elastic`: launch distributed training, if errors are expected or if resources can join and leave dynamically (multiple tasks run on a finite set of clusters).

The necessary `torch` modules are:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import DataParallel
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributeSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
```

## Data Parallelism

The parallelism behind Data Parallelism (DP) is that the model is replicated to all GPUs, in which each GPU consumes different partition of the input data. The constraint on this paradigm is strongly related to the fact, *DataParallel replicates the same model to all GPUs*, resulting in an impossible mission for large models that cannot fit in one GPU.

```python
device = torch.device("cuda:0")

class DataParallelModel(nn.Module):
    ...

model = DataParallelModel()
if torch.cuda.device_count() > 1:
      model = DataParallel(model)
model.to(device)

for data in SOME_DATA_LOADER:
    inputs = data.to(device)
    output = model(inputs)
    # the size of tensors tends to be evenly distributed
    # e.g., torch.Size([30, 5]) on 8 GPUs results in
    # 7 torch.Size([4, 5]) and 1 torch.Size([2, 5])
    print(f'{inputs.size()=} and {output.size()=}')
```

DP automatically splits data and sends jobs to multiple models, that is replicated on several GPUs. It finalizes the communication by collecting and merging the results before actual returning. The following manual drawing chart demonstrates how DP is executed via PyTorch.

![[Pasted image 20241107153043.png]]

> by [Sebastian Raschka](https://x.com/rasbt/status/1570442488334397440)

> [!todo] Use case
> Single-process, multi-thread, and only works on a single machine for a idea validation that demands less code modification in a short period of time.

> [!warning] Limitation
>
> - The model has to be small enough to fit in one device.
> - Parallelism is limited by batch size, as too much data leads to inefficient steps.
> - Most importantly, it has been proven that DistributedDataParallel is faster for a single-node multi-gpu task.

## Distributed Data Parallelism

Distributed Data Parallelism (DDP) supports multi-process and multi-node. The idea of DDP behind the hood is spawning multiple processes and create a single instance of DDP exclusively per process.

### Basic Use

#### Initialize and Destroy DDP

A PyTorch DDP needs to be initialized properly before collaborating your model and data parallel, and it also needs to be destroyed after. The initialization guaranteed all the process are blocked until they all join.

```python
import os

# "gloo" - distributed CPU training
# "nccl" - distributed GPU trianing with CUDA drivers
# "mpi"  - last probable choice
DIST_BACKEND = "nccl"

def initialize(rank, world_size, timeout=None):
    # don't worry about how to access rank for each process just yet
    # or take a glimpse at #Spawn-Subprocess
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # ensure that each process exclusively works on a single GPU from
    # 0 to world_size-1, i.e., their global rank (or just rank)
    torch.cuda.set_device(rank)  # alternative to set CUDA_VISIBLE_DEVICE
    init_process_group(DIST_BACKEND, rank=rank, world_size=world_size, timeout=timeout)
    
def destroy():
    destroy_process_group()
```

The `initialize` method will be invoked by spawning subprocesses in the `torch.multiprocessing` module, so that each subprocess exclusively works on a single GPU.

#### Distributed DataLoader

PyTorch provides a distributed version of sampler, `torch.utils.data.distributed.DistributedSampler` in conjunction with `torch.nn.parallel.DistributedDataParallel`, in which each process is provided with a `DistribuetdSampler` instance as a `DataLoader` sampler.

```python
def prep_dataloader(dataset, batch_size):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
    )
```

In order to ensure all replicas use a different random ordering for each epoch, `set_epoch()` needs setting for each epoch. E.g.,

```python
train_data = prep_dataloader(...)

def train(...):
    for epoch in range(max_epochs):
        train_data.sampler.set_epoch(epoch)
    ...
```

#### Train Model with DDP

A simple example for a model, that fits in one GPU, wrapped with `DistributedDataParallel` model class.

```python
class DataParallelModel(nn.Module):
  """A user-defined model fit in one GPU
      """
  ...

def train(train_data, rank, max_epoch, snapshot):
    model = DataParallelModel().to(rank)
    
    # device_ids is an optional argument for class `DistributedDataParallel`
    # as spec says, a device_ids can either be None or List of single integer
    # regarding to two scenarios:
    # 1) for single-device modules, device_ids can contain exactly one device
    # id, e.g., device_ids = [0];
    # 2) for multi-device modules, device_ids must be None.
    # See official documentation for more information.
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    for epoch in range(max_epochs):
        train_data.sampler.set_epoch(epoch)
        for src, tgt in train_data:
            src = src.to(rank)
            tgt = tgt.to(rank)
            optimizer.zero_grad()
            out = ddp_model(src)
            loss = F.cross_entropy(out, tgt)
            loss.backward()
            optimizer.step()
        
        if rank == 0 and epoch % snapshot == 0:
            # wait for its definition later on
            save_checkpoint(ddp_model)
```

Please be rest assured that gradients are synchronized by the communication throughout GPUs during the backward pass and overlap with the backward computation. `backward()` triggers the internal hook of synchronization, and thus `param.grad` already contains the updated gradient tensors.

> [!note]
> A further note for synchronization is unnecessary for manually calling synchronization primitives (e.g., `dist.barryer()`) after returning from a `backward()`.

However, each processes may experience different synchronization stages (i.e., constructor, the forward pass and the backward pass). A terrible scenario, where fast processes arrive early and timeout while waiting for others, could happen. Therefore, it is the users' responsibility that maintain the balance of workloads across processes. Sometime, a large `timeout` in initialization might be a good practice (`timeout` is 10 minutes for `nccl` backend by default).

### Spawn Subprocess

Since we have initialized `ProcessGroup`, prepared `DataLoader` and `train` script, the next step is to spawn up multiple processes per node, in which each process uniquely utilizes one GPU. A sample `main` entry is demonstrated as follows.

```python
def main(rank, world_size, dataset, snapshot, max_epochs, batch_size):
    initialize(rank, world_size)
    train_data = prep_dataloader(dataset, batch_size)
    train(train_data, rank, max_epoch, snapshot)
    destroy()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Training with PyTorch DistributedDataParallel Module.")
    parser.add_argument("--world-size", type=int, help="World Size")
    parser.add_argument("--snapshot", type=int, help="How often saving a snapshot")
    parser.add_argument("--max-expoch", type=int, help="Maximum epochs for training")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size on each device (default: 32)")
    args = parser.parse_args()
    
    mp.spawn(main,
             args=(args.world_size,
                   dataset,
                   args.snapshot,
                   args.max_epochs,
                   args.batch_size,
             ),
             nprocs=world_size
    )
```

The driver code is relative simple and straightforward. However, there are few pitfalls listed as follows,

- The `main` method requires six positional arguments while we only provide five positional arguments for multiprocessing to spawn the method. The reason is that `multiprocessing.spawn` invokes the function `fn` (i.e., `main` above) by calling `fn(i, *args)`, where `i` is the process index (**THIS IS HOW WE GET GLOBAL RANK**) and `args` is the passed through arguments. In other words, `torch` drivers automatically generate the rank of each GPU (or process id) and pass it back. Other than that, `torch` also provides an API, `torch.distributed.get_rank()`, to directly access the rank.
- The way of Python `multiprocessing` works is by staring *n* Python processes, importing the main module, and then calling your specified entry point (i.e., `fn`). Supposing, if there is no main guard (i.e., `if __name__ == "__main__"`), *n* Python processes each will execute the code line-by-line from top to bottom. As a result, subprocesses will keep on creating more of them recursively, and raise a `RuntimeError`. It is also refers to [Safe importing of main module](https://docs.python.org/3.10/library/multiprocessing.html#multiprocessing-safe-main-import) in the multiprocessing guidelines. [PyTorch official FAQ also mentioned](https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection).

<iframe src="https://embed.reddit.com/r/learnpython/comments/tc21ah/comment/i0ayz65/?embed=true&amp;ref_source=embed&amp;ref=share&amp;utm_medium=widgets&amp;utm_source=embedv2&amp;utm_term=23&amp;showmedia=false&amp;showmore=false&amp;depth=1&amp;utm_name=comment_embed&amp;embed_host_url=https%3A%2F%2Fpublish.reddit.com%2Fembed" width="640" scrolling="no" allowfullscreen="true" sandbox="allow-scripts allow-same-origin allow-popups" allow="clipboard-read; clipboard-write" style="border: none; max-width: 100%; border-radius: 8px; display: block; margin: 0px auto;" height="340"></iframe>

### Elastic Launch

Elastic launch (or `torchrun`) is a superset of `torch.distributed.launch`, providing additional functionalities such as:

1. Handle worker failures by restarting all workers,
2. Assign worker `RANK` and `WORLD_SIZE` automatically,
3. and elastic adjust the number of nodes according to needs.

When scaling nodes, both scale-down and scale-up will stop training until a new `WorkerGroup` is formed along with new `RANK` and `WORLD_SIZE`. Thus, DO NOT hard code any assumptions about `RANK`, using **environment variable** given by `torch` drivers instead. The same thing applies to `WORLD_SIZE` as well.

The `torchrun` launcher starts the [agent process](https://github.com/pytorch/pytorch/blob/e6ff07f00e04a9b58efb86a3dd70ed7280ae8522/torch/distributed/elastic/agent/server/local_elastic_agent.py#L291), and then sets and passes [local rank](https://github.com/pytorch/pytorch/blob/e6ff07f00e04a9b58efb86a3dd70ed7280ae8522/torch/distributed/run.py#L871) and [world size](https://github.com/pytorch/pytorch/blob/e6ff07f00e04a9b58efb86a3dd70ed7280ae8522/torch/distributed/run.py#L790) to users by [parsing arguments and previous configurations](https://github.com/pytorch/pytorch/blob/e6ff07f00e04a9b58efb86a3dd70ed7280ae8522/torch/distributed/run.py#L774). All available environment variables (e.g., `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, etc) set by launcher are listed [here](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).

#### Sample Training Script

A sample training script with recommended structure is demonstrated as follows,

```python
# file: train_script.py
import os
from torch.distributed.elastic.multiprocessing.errors import record

def initialize():
    # correct use of rank, provided by launcher
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # make sure each process exclusively operates on a dedicated GPU
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    model = MyModel().to(local_rank)
    ddp_model = DDP(model, 
                    device_ids=[local_rank],
                    output_device=local_rank)
    ...

# add `@record` decorator to enable summary on worker errors,
# including time, rank, pid, traceback, etc.
@record
def main():
    load_checkpoint(...)
    initialize(...)
    train(...)

if __name__ == "__main__":  # always guard your entrypoint before execution
    main()
```

#### Run Training Script

```shell
# file: multinode.sh
# Get master address from some task scheduling tools, 
# SLURM for example
$ export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)


EALSTIC=true
if [[ $ELASTIC = true ]]; then
  # an elastic number of workers
  NNODES=1:4
else
  # a fixed number of workers
  NNODES=$(nvidia-smi --list-gpus | wc -l)
fi

$ torchrun --nnodes=${NNODES} \
           --nproc_per_node=${NGPU_PER_NODE} \
           --node_rank=0 \  # scales up for multi-node use case
           --max-restarts=3 \  # tolerates up to 3 failures and restart all
           --rdzv_id=${JOB_ID} \
           --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
           <train_script.py> <--args>
```

Note, users have to make sure that each instance is setup on different ports to avoid port conflicts.

### Deployment

#### Single-node Multi-worker

Start the launcher on the host to start the agent process which creates and monitors a local work group under the hood.

#### Multi-node Multi-worker

Start the launcher with the same arguments on all the nodes participating in training.

### Activation Checkpointing

Checkpointing can also gain improvements from applying parallelism. As all processes start with the same parameters and optimizers set the parameters to the same value (i.e., merge, or **gather** to be precise, results from all processes by calling `backward()`), there is no necessity saving `state_dict` from each processes nor loading $n$ many times.

The implementation thus is optimized by saving model's `state_dict` from one process. And each processes halt until the `save` operation is done. Subsequently, each process can specify the way of loading `state_dict` by `map_location`.

> [!note]
> Without specifying `map_location`, tensors are going to be loaded to CPU and then to the device where it was saved (e.g., process[0] in the following example).

```python
import tempfile

def save_checkpoint(ddp_model)
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.pt"
    if rank == 0:
        # some code refers state_dict of DDP models to the following,
        # ddp_model.module.state_dict() because module is an attribute
        # but technically, `DDP` is a subclass of `nn.Module`
        # hence directly access state_dict should be alright
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # block all processes until process[0] (i.e., rank 0) finishes saving
    dist.barrier()

    # specify `map_location` to justify each process
    map_location = {"cuda:0": f"cuda:{rank}"}
    ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH
                       map_location=map_location))

    # no `barrier()` needed as explained in #Wrap-Model-with-DDP
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    destroy()
```

## Model Parallelism

As noted in the first section, DP requires replicating the model on each device, resulting in redundant memory consumption issue when training large models. To address this limitation and improve memory efficiency, approaches such as Model Parallelism (MP) and Pipeline Parallelism (PP) have been developed.

MP splits the model vertically (i.e., across multiple layers), partitioning the computation and parameters in each layer across multiple devices, requiring significant communication between each layer [4]. It can be seen that, due to the high bandwidth of inter-GPU communication within a single node, data moved back and forth across GPUs in parallel do not compromise the performance (i.e., the [memory overhead](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/#overhead) is hidden by fast inter-GPU communication) but speed up the collective process instead. However, the degradation comes up when scaling nodes. The constraint of this parallelism mainly lies in the inefficient communication between multi-nodes. In a brief, compared to DP, MP partitioning the parameter does obtain high memory efficiency, but it is less scaling efficient.

![[Pasted image 20241107153506.png]]

> by [https://docs.chainer.org/en/v7.8.0/_images/parallelism.png…](https://t.co/JsXK6tOrYG)

DDP can work with MP, where each process would use MP, and all processes collectively would use DDP.

```python
class ModelParallelModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
        ).to("cuda:0")
        self.seq2 = nn.Sequential(
            nn.Conv2d(20, 64, 5),
            nn.ReLU(),
        ).to("cuda:1")
        self.fc = nn.Linear().to("cuda:1")

    def forward(self, x):
        # the results of seq1 needs to be moved to "cuda:1"
        # for seq2 to be computed
        x = self.seq2(self.seq1(x).to("cuda:1"))
        return self.fc(x.view(x.size(0), -1))
```

### Tensor Parallel

Tensor parallelism is a further fine-grained parallelism used to fit a large model into multiple GPUs.

![[Tensor Parallel.png]]

Take one MLP block within transformers as example. It only contains a nonlinearity GeLU function after a GEMM of $XA$.

$$
Y = \mathrm{GeLU}(XA)
$$

An option to parallelize the GEMM is to split to weight matrix $A$ horizontally (along its rows) as well as input $X$ vertically (along its columns).

$$
X = \begin{bmatrix}X_1, X_2\end{bmatrix},\ 
A = \begin{bmatrix}A_1\\A_2\end{bmatrix}
$$

Consequently, the computation now becomes,

$$
\begin{align*}
Y=\mathrm{GeLU}(XA)&=\mathrm{GeLU}(X_1A_1+X_2A_2)\\
&\ne \mathrm{GeLU}(X_1A_1)+\mathrm{GeLU}(X_2A_2)
\end{align*}
$$

## Pipeline Parallelism

Pipeline Parallelism (PP) horizontally splits the model across layers running each partition on a different device and use micro-batching to hide the pipeline bubble [4].

> [!note]
> Model parallel requires copying data back and forth across GPUs, so that overhead exists. One improved implementation would be pipelining, as in once the `seq1` finishes the computation and finalizes the data movement, it can start with the second micro-batch.

![[Pasted image 20241107153855.png]]

> by [https://fairscale.readthedocs.io/en/latest/_images/pipe.png…](https://t.co/LHMUUK1rfJ)

```python
class PipelineParallelModel(ModelParallelModel):
    def __init__(self, split_size=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # split_size control the granularity of pipelining
        # the small split size is, the fine pipelining gets
        # hence less overhead
        self.split_size = split_size

    def forward(self, x):
        chunk = iter(x.split(self.split_size))
        cnext = next(chunk)
        # first computation happened on GPU:0 and moved to GPU:1
        cprev = self.seq1(cnext).to("cuda:1")
        ret = []

        for cnext in splits:
            # finish computation on GPU:1
            cprev = self.seq2(cprev)
            ret.append(self.fc(cprev))
            # repeat first computation
            cprev = self.seq1(cnext).to("cuda:1")

        cprev = self.seq2(cprev)
        ret.append(self.fc(cprev))

        return torch.cat(ret)
```

> [!quote] Comparison between two model parallel strategies.
> ![[Pasted image 20240703145522.png]]

## Fully Sharded Data Parallel

Fully Sharded Data Parallel (FSDP)


---

### Reference

[1] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.” arXiv, Mar. 13, 2020. Accessed: Jul. 01, 2024. [Online]. Available: [http://arxiv.org/abs/1909.08053](http://arxiv.org/abs/1909.08053)
[2] torchrun (Elastic Launch) — PyTorch 2.5 documentation. URL: https://pytorch.org/docs/stable/elastic/run.html
[3]  GitHub ddp-tutorial-series distributed-pytorch at https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
[4] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,” May 13, 2020, _arXiv_: arXiv:1910.02054. Accessed: Nov. 05, 2024. [Online]. Available: [http://arxiv.org/abs/1910.02054](http://arxiv.org/abs/1910.02054)