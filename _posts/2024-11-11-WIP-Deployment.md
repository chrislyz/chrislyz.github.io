---
layout: post
---

## Llama.cpp

## Ollama

## vLLM

### Offline Inference Engine

```python
from vllm import LLM, SamplingParams

prompts: List[str] = [

]

# Models are downloaded from HuggingFace by default,
# set env variable VLLM_USE_MODELSCOPE=True otherwise


# Full arguments is listed here
# https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html#llm-class
llm = LLM(model="facebook/opt-125m"
          tokenizer="NAME_OF_TOKENIZER")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
```

### Serving with Local OpenAI-Compatible Server

To begin with, we start the server at `http://localhost:8000` by default.

```shell
vllm serve <model_name> [--host HOST] [--port PORT]
```

> [!note]
> [Full command line arguments are listed here](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#command-line-arguments-for-the-server)


Once the server is started, it can be queried through an API in an OpenAI-like format,

```shell
curl --location http://localhost:8000/v1/<some/task> \
    [--api-key API-KEY] \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer token' \
    --data '{
        "model": <"model-name">,
        "messages": [
            {
                "role": "user",
                "content": <"some-content-here">
            }
        ]
    }'
```

### Serving with Docker

```shell
export HF_TOKEN=<hf-access-token>
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model <model_name>
```

### Serving with Kubernetes

#### An Introduction of Kubernetes Management

Use `kubectl apply` to create all objects, except for existing ones, defined by `yaml` configurations in a specified directory:

```shell
# create all objects to a directory with all configurations
# [-R] option applies recursively
kubectl apply -f [-R] <directory>
```

Use `kubectl apply` to update all existing objects:

```shell
kubectl diff -f [-R] <directory>    # sets fields in living config
kubectl apply -f [-R] <directory>   # clears fields
```

Alternatively, one can directly update some fields by `kubectl sclae`:

```shell
# kubectl scale <filename|url> <--field=value>
# example:
kubectl scale deployment/vllm_deployment_with_k8s --replicas=2
```

Print living configuration using `kubectl get`:

```shell
# kubectl get -f <filename|url> -o yaml
# example:
kubectl get deployment/vllm_deployment_with_k8s -o yaml
```

Delete objects using either `kubectl delete` or `kubectl apply`:

```shell
kubectl delete -f <file-name>

# removes objects whose manifest have been removed
# similar to "clean" command in some package manager
kubectl apply -f <directory> --prune
```

> [!note]
> [Thorough and comprehend introduction](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/declarative-config/)

Now, let us come back to deploy vLLM with K8s. The deployment requires three steps listed as follows,

#### 1. Create a PVC, Secret and Deployment

Create a [PVC (Persistent Volume Claim)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) object to store the model cache. Optionally, you can use [hostPath](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath) or [other storage](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#types-of-persistent-volumes) choices. The manifest is described in the following:

```yaml
# pvc.yaml
# See more about PVC in
# https://kubernetes.io/docs/concepts/storage/persistent-volumes/
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: <name-of-storage>
  namespace: default
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: default
  volumeMode: Filesystem
```

and

```shell
kubectl apply -f ./pvc.yaml
```

Create a [Secret](https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-config-file/) object only for accessing gated models by providing `"SECRET_AUTH_TOKENS`. The manifest is describe in the following:

```yaml
# See more about secret manifest in
# https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-config-file/
# opaque_secret.yaml
apiVersion: v1
kind: secret
metadata:
  name: hf-token-secret
  namespace: default
type: Opaque
data:
  token: <"SECRET_AUTH_TOKENS">
```

and

```shell
kubectl apply -f ./opaque_secret.yaml
#$: secret/opaque_secret configured

kubectl get secret opaque_secret -o yaml  # returns in yaml format
```

Create a [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) object to run the application on K8s clusters. Meanwhile, it manages stateless application workloads. The manifest is described in the following with omitting part of the configuration:

```yaml
# vllm_deployment_with_k8s.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: <name-of-deployment>
  namespace: default
  labels:
    app: <app-name>
    tier: <tier-name>  # for large applications with multiple tiers
spec:
  replicas: 1
  selector:
    matchLabels:
      app: <app-name>
...
# For detailed spec, visit
# https://docs.vllm.ai/en/stable/serving/deploying_with_k8s.html
```

and

```shell
kubectl apply -f ./vllm_depolyment_with_k8s.yaml

kubectl get deployments  # check if Deployment was created
kubectl get pods -l run=app_name -o wide
```

#### 2. Create a K8s Service

Once we have succeed in deploying vLLM into k8s pods locally, we want to expose the application running in clusters so that clients can interact with it. Create a [Service](https://kubernetes.io/docs/concepts/services-networking/service/) object to expose our application. The manifest is describe in the following,

```yaml
# vllm_svc.yaml
apiVersion: v1
kind: Service
metadata:
  name: <service_name>
  namespace: default
spec:
  ports:
  - name: <name-of-service-port>
    port: 80
    protocol: TCP
    targetPort: 8000
  selector:
    app: <app-name>   # match the deployment labels/app
    tier: <tier-name> # match the deployment labels/tier
  sessionAffinity: None
  type: ClusterIP
```

```shell
kubectl apply -f ./vllm_svc.yaml

kubectl get svc app_name
kubectl describe svc app_name
```

#### 3. Test the Service

```shell
curl http://app_name.default.svc.cluster.local/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": ["model_name"],
        "prompt": ["prompts"],
        "max_tokens": [max_tokens],
        "temperature": 0
     }'
```

### Distributed Inference and Serving

Generally, one can decide the distributed inference strategy by the spec of possessed hardwares such as:

- **Single GPU**, if your model fit in one GPU entirely;
- **Single-Node Multi-GPU**, if your model is too large to fit in one GPU but in multiple GPUs, where the total video memory is determined by the GPU memory (per device) multiplied by the maximum number of GPUs supported in a single node (denoted as *Server Options* in the GPU spec, e.g., A100 can be grouped with up to 8 GPUs resulting );
- **Multi-Node Multi-GPU**, otherwise.

Essentially, we want to perform inference on a distributed system by scaling with platform until certain hardware specification is met, in which case adding enough GPUs and nodes to hold the model.

#### Parallelisms

See more about [[Distributed Training#Tensor Parallel]] and [[Distributed Training#PipelineParallel (Inter-Layer)]] in detail.

```shell
vllm serve <model-name> \
    [--tensor-parallel-size size] \
    [--pipeline-parapllel-size size]
```

- `tensor-parallel-size` is defined by the number of GPUs (or devices) on each node;
  - `total_num_attention_heads` has to be a multiple of `tensor-parallel-size`;
  - and if `model_size` cannot be divided by `num_of_gpus`, use pipeline parallel instead;
- `pipeline-parallel-size` is defined by the number of nodes in a cluster;
- `vllm/vllm/config.py:894` defines the `world_size` by the product of `pipeline_parallel_size` and `tensor_parallel_size`;

#### Multi-Node Inference and Serving

vLLM provided a helper [script](https://github.com/vllm-project/vllm/blob/main/examples/run_cluster.sh) to start the Ray cluster in which the script sets up head or worker nodes with given docker images and configurations to ensure the exact same environment as well as to hide heterogeneity of the host machines.

```shell
bash run_cluster.sh \
    <DOCKER_IMAGE> \
    <HEAD_NODE_ADDRESS> \
    <NODE_TYPE> \
    <PATH_TO_HF_HOME>
```

Subsequently, we serve vLLM as usual,

```shell
vllm serve /path/to/model/in/container\
    [--tensor-parallel-size size] \
    [--pipeline-parallel-size size]
```

> [!note]
> Performance parallelisms requires efficient communications among nodes. Address such using high-speed network cards like Infiniband (IB).

## Megatron

## Deepspeed

## TensorRT-LLM

## HuggingFace Text Generation Inference

## RayLLM

## Triton Inference Server