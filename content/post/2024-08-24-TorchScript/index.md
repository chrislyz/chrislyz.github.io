---
title: TorchScript
date: 2024-08-24
---

### Rationale

Although PyTorch works like a charm and provides an enormous number of features of creating machine learning models, deploying such models into a production environment with requiring high-performance is painful. TorchScript is hence developed to create serializable and optimizable models from PyTorch code. TorchScript program can be run independently from Python, such as in a standalone C++ program. This makes it possible to train the model in a development environment with Python and export the model via TorchScript to a production environment where a language with low-level accessibility and high-performance requirement is necessary.

### Basics of TorchScript

#### Tracing Modules

Supposing we have a custom network module defined as,

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        ...
    def forward(self, x):
        ...
```

TorchScript provides tools to capture the definition of our models by *tracing*.

```python
net = MyNetwork()
example_inp = torch.rand(3, 4)
traced_cell = torch.jit.trace(net, (example_inp,))
```

The return of a `trace` depends on the `Callable` input `func`. If `func` is `nn.Module` or `forward` of `nn.Module`, `trace` returns a `ScriptModule` object. Otherwise, it returns a `ScriptFunction`.

TorchScript records the definition of a *nn.Module* in an Intermediate Representation (or computation graph). One can examine the graph with its `graph` property or even higher interpretation in Python syntax with `code` property

```python
# output low-level computation graph representation
traced_cell.graph

# output high-level human reable Python-syntax representation
traced_cell.code
```

> [!warning] WARNING
> Tracing only correctly records functions and modules which are **not data dependent**. Any control flow statements and external dependencies (e.g., access global variables) are erased. However, there is one exception when this control-flow is constant across the module (guess due to the reason of easy-to-optimize)

`TorchScript` also provides a **script compiler** to record functions and modules even bundled with control flow. The script compiler does direct analysis of Python source code to transform it into TorchScript.

```python
scripted_cell = torch.jit.script(net)
scripted_cell.code
```

#### Saving and Loading Models

`TorchScript` provides APIs to save and load modules to/from disk in an archive format.

```python
traced_cell.save('jit_traced_cell.pt')

jit_model = torch.jit.load('jit_traced_cell.pt')
```

### Loading a TorchScript Model in C++

#### Converting PyTorch Model to TorchScript Module

A simple approach to convert your PyTorch model to TorchScript module via [[#Tracing Modules]].

Any methods in **nn.Module** can be excluded with the annotation of `@torch.jit.ignore` decorator.

#### Serializing ScriptModule to File

At this step, we want to export the model to be language independent.

```python
traced_cell.save('jit_traced_cell.pt')
```

> [!Note] THIS SERIALIZED FILE **IS NOT** A CHECKPOINT FILE

#### Loading Script from Other Languages

Take C++ as example. To be able to load the serialized model in C++, PyTorch C++ API - also known as [*LibTorch*](https://pytorch.org/get-started/locally/) - is a necessity.

```cpp
#include <torch/script.h>
#include <iostream>
#include <vector>

torch::jit::script::Module module;

bool LoadScriptModule(const char* module_path) {
    try {
        // Deserialize the ScriptModule from the File
        module = torch::jit::load(module_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the module\n";
        return false;
    }
    return true;
}
```

#### Executing ScriptModule

Since then, we have left the boundary of Python. All PyTorch operations should strictly abide by [TorchScript reference](https://pytorch.org/docs/master/jit.html) and [PyTorch C++ API documentation](https://pytorch.org/cppdocs/).

```cpp
void predict(void) {
    std::vector<torch::jit::IValue> inputs(torch::ones({1, 3, 224, 224}));
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(1, 0, 5) << '\n';
}
```

### Best Practice

First thing first, it is a common confusion about, "If we have scripting that covers all most all of the cases, why do we still need tracing?". The answer is that scripting compromises the flexibility and quality of the code it generated.

#### Type Annotation

According to the [official documentation](https://pytorch.org/docs/master/jit_language_reference_v2.html#when-to-annotate-types), type annotations are necessary when static types cannot be automatically inferred. However, the real-word case is often more complicated rather than the type automatically inferred. For instance, sometimes an inferred type can be too restrictive to the literal a user instantiated. A assignment of `x = None` could overwrite the potential `Optional` type to a more restrictive `None` type.

TorchScript presumes any parameter, local variable, or data attribute that is neither type annotated nor able to be inferred automatically, to be one of `TensorType, List[TensorType]` or `Dict[str, TensorType]`. Therefore, for the reason of safety and bug-free, it is recommended to annotate functions and variables especially in multi-types (e.g., `Union[str, int]`), and thus treat Python as statically typed languages.

##### Annotate Functions

```python
Python3Annotation ::= "def" Identifier ["(" ParamAnnot* ")"] [ReturnAnnot] ":"
    MethodBody
ParamAnnot  ::= Identifier [":" Type] ","
ReturnAnnot ::= "->" Type

# e.g.,
def f(a: torch.Tensor, b: int) -> torch.Tensor:
    return a + b
```

##### Annotate Variables

```python
LocalVarAnnotation ::= Identifier [":" Type] "=" Expr

# e.g.,
value: Optional[torch.Tensor] = None
```

##### Annotate Instance Data Attributes

```python
"class" ClassIdentifier "(torch.nn.Module):"

InstanceAttrIdentifier ":" ["Final("] Type [")"]

# e.g.,
class MyModule(torch.nn.Module):
    BATCH_SIZE: Final[int]
```

#### Type Annotation APIs

```python
@torch.jit.script
def fn():
    d = torch.jit.annotate(Dict[str, int], {})

    # Without jit.annotate, TorchScript will automatically infer
    # d as Dict[str, torch.Tensor], hence result in error "type
    # mismatch"
    d["name"] = 20
```

#### Limitations

The user-defined module has to be a proper connected graph representable in `TSType` (read more in [here](https://pytorch.org/docs/master/jit_language_reference_v2.html#type-annotation-appendix)). Other than that, both of `trace` and `script` only record Python and PyTorch operations. In other words, using of `numpy.ndarray` and invoking `OpenCV` APIs will not be recorded in either way, and thus it requires code refactoring.

##### Aggressive Optimization

The compiler usually tends to aggressively optimize the static evaluation to a constant and fails to generalize. A typical example would be getting tensor shape with Python builtin method `len`.

```python
a, b = torch.rand(1), torch.rand(2)
def f1(x): return torch.arange(x.shape[0])
def f2(x): return torch.arange(len(x))
torch.jit.trace(f1, a)(b)  # output: tensor([0, 1])
traced = torch.jit.trace(f2, a)
traced.code
# output: 'def f2(x: Tensor) -> Tensor:\n  _0 = torch.arange(1, dtype=None, layout=None, device=torch.device("cpu"), pin_memory=False)\n  return _0\n'
traced(b)
# output: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
# tensor([0])
```

Since the value of $a$ and $b$ are deterministic, the compiler aggressively optimizes the length of the input tensor in `f2()` to a constant shown by output of `traced.code` in the line which says `_0 = torch.arange(1, dtype=None, ...)`. As a result, the traced model will fail if we feed with a tensor of a different shape.

The similar failure happens to,

- `torch.item()` that converts a scalar in tensor format to Python numbers (i.e., `int` or `float`),
- methods that convert `torch` types to Python, `numpy`, or other library primitives,
- slicing and advanced indexing,
- `device()` that sets device that actually stores and computes the data. Note that this method **will not throw** any warnings.

#### Condition Flow

##### Hard Refactor

We have mentioned that tracing modules will not correctly record the control flow but simply code path from one of the branch. In order to write clean code, it is a good practice to put underlying control flow into the kernel implementation. Finally, tracing such modules become available.

##### Soft Refactor

By all means, users want to refactor the code with minimum modification. Scripting is allowed to combine with tracing. As long as we let scripting handle control flow, the rest can be safely handled to tracing. `@script_if_tracing` decorator is such a convenient tool to script only the necessary block of implementation. For example,

```python
def forward(self, ...):
    # ... some forward logic

    @toch.jit.script_if_tracing
    def _inner_impl(...):
        # implment control flow here
        return ...

    result = _inner_impl(...)

    # ... other forward logic
```


#### Compatibility Test

As we discover many precautions to prevent TorchScript from generalizing a different model than expected, it is unreliable to depend on only manual examination. There is a easy to unittest the compatibility between the model we designed and the model compiled between TorchScript by testing the different between outputs of the two models.

```python
traced_model = torch.jit.trace(model, input1)
assert allclose(traced_model(input2), my_model(input2))
```

---

### Reference

[1] Yuxin Wu's Blog: [TorchScript: Tracing vs. Scripting](https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/)