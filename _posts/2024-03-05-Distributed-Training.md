---
layout: post
---



## Distributed Data-Parallel Training

Distributed Data-Parallel Training (DDP) is a paradigm adopting the single-program multiple-data paradigm. DDP replicate the model in each process but feed them with different samples. Gradients are synchronized throughout the communications.

## Overview

PyTorch provides developers with 6 ways of training paradigms:

1. `torch.nn.DataParallel`: Single-device, if data and model can fit in **one** GPU,
2. `torch.nn.DataParallel`: Single-machine multi-GPU to make use of multiple GPUs to speed up training with minimal change of codes,
3. `torch.nn.parallel.DistributedDataParallel`: Single-machine multi-GPU to further speed up training with little change of codes,
4. `torch.nn.parallel.DistributedDataParallel`: Multi-machine launching script to scale up across machines,
5. `torch.distributed.FullyShardedDataParallel`: Multi-GPU training on a single-machine or multi-machine, if the data and model can not fit in one GPU,
6. `torch.distributed.elastic`: launch distributed training, if errors are expected or if resources can join and leave dynamically (multiple tasks run on a finite set of clusters).

> [!tip] Terms
> - Node: An end or a machine,
> - World Size: The number of processes participating in the job, in which each process occupies a GPU. Hence, The total number of processes is equivalent to the number of GPUs, `world_size = torch.cuda.device_count()`,
>     - $\textrm{world\_size} = \textrm{n\_nodes} \times \textrm{ngpu\_per\_node}$
> - Rank: A unique identifier assigned to each process within a distributed process group. They are always consecutive integers ranging from 0 to `world_size`,
> - Local Rank: The rank (id) of the process on the local machine.

## Data Parallel

The parallelism behind DataParallel is that the model is replicated to all GPUs, in which each GPU consumes different partition of the input data. The constraint on this paradigm is strongly related to the fact, *DataParallel replicates the same model to all GPUs*, resulting in an impossible mission for large models that cannot fit in one GPU.

```python
from torch.nn import DataParallel

device = torch.device("cuda:0")

class DataParallelModel(nn.Module):
    ...

model = DataParallelModel()
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model.to(device)

for data in SOME_DATA_LOADER:
    input = data.to(device)
    output = model(input)
    # the size of tensors tends to be evenly distributed
    # e.g., torch.Size([30, 5]) on 8 GPUs results in
    # 7 torch.Size([4, 5]) and 1 torch.Size([2, 5])
    print(f'{input.size()=} and {output.size()=}')
```

DataParallel automatically splits data and sends jobs to multiple models, that is replicated on several GPUs. It finalizes the communication by collecting and merging the results before actual returning.

> [!todo] Scenario
> Single-process, multi-thread, and only works on a single machine.

## Distributed Data Parallel

### Basic Use

#### Initialize and Destroy DDP

A PyTorch DDP needs to be initialized properly before collaborating your model and data parallel, and it also needs to be destroyed after. The initialization guaranteed all the process are blocked until they all join.

```python
import os
import torch.distributed as dist

# "gloo" - CPU training
# "nccl" - GPU trianing with CUDA drivers
# "mpi" - GPU trianing without CUDA drivers
DIST_BACKEND = "nccl"

def initialize(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(DIST_BACKEND, rank=rank, world_size=world_size, timeout=timeout)
    

def destroy():
    dist.destroy_process_group()
```

#### Wrap Model with DDP

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
             
class MyModel(nn.Module):
    ...

def train(rank, world_size):
    initialize(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))

    # labels needs to be moved to the same GPU for backprop
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    destroy()
```

Please be rest assured that gradients are synchronized by the communication throughout GPUs during the backward pass and overlap with the backward computation. `backward()` triggers the internal hook of synchronization, and thus `param.grad` already contains the updated gradient tensors.

> [!note]
> A further note for synchronization is unnecessary for manually calling synchronization primitives (e.g., `dist.barryer()`) after returning from a `backward()`.

However, each processes may experience different synchronization stages (i.e., constructor, the forward pass and the backward pass). A terrible scenario, where fast processes arrive early and timeout while waiting for others, could happen. Therefore, it is the users' responsibility that maintain the balance of workloads across processes. Sometime, a large `timeout` in initialization might be a good practice (`timeout` is 10 minutes for nccl backend by default).

#### Checkpoint

Checkpointing can also gain improvements from applying parallelism. As all processes start with the same parameters and optimizers set the parameters to the same value (i.e., merge, or **gather** to be precise, results from all processes by calling `backward()`), there is no necessity saving `state_dict` from each processes nor loading $n$ many times.

The implementation thus is optimized by saving model's `state_dict` from one process. And each processes halt until the `save` operation is done. Subsequently, each process can specify the way of loading `state_dict` by `map_location`.

> [!note]
> Without specifying `map_location`, tensors are going to be loaded to CPU and then to the device where it was saved (e.g., process[0] in the following example).

```python
import tempfile

def train():
    initialize(rank, world_size)
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # block all processes until process[0] finishes saving
    dist.barrier()

    # specify `map_location` to justify each process
    map_location = {"cuda:0": f"cuda:{rank}"}
    ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH
                       map_location=map_location))

    # backward pass
    ...

    # no `barrier()` needed as explained in #Wrap-Model-with-DDP
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    destroy()
```

#### ModelParallel

DDP can work with model parallel, where each process would use model parallel, and all processes collectively would use data parallel.

Model parallel is an idea of parallelism that rather splitting the model (i.e., multiple layers) across different GPUs, especially for large models that do not fit in one GPU.

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

<br/>

##### PipelineParallel

> [!note]
> Model parallel requires copying data back and forth across GPUs, so that overhead exists. One improved implementation would be pipelining, as in once the `seq1` finishes the computation and finalizes the data movement, it can start with the second batch.

![]({{site.baseurl}}/assets/media/Pasted image 20240530114508.png)

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

#### TorchRun

```python
def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # e.g., supposing we have 2 nodes and 4 gpus each, 8 in total
    #            node 0       |      node 1
    # rank | [0],[1],[2],[3]  | [4],[5],[6],[7]
    # hence, device_id
    device_id = rank % torch.cuda.device_count()
    model = MyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
   
```

Command to initialize DDP jobs on all nodes is then

```shell
# Get master address from some task scheduling tools, 
# SLURM for example
$ export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

$ torchrun --nnodes=$(nmachines) \
           --nproc_per_node=$(ngpu_per_machine) \
           --rdzv_id=100 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:29400 \
           ddp.py
```

![]({{site.baseurl}}/assets/media/Pasted image 20240611164616.png)

## Fully Sharded Data Parallel

## Tensor Parallel