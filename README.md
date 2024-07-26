# Slower - Split Learning Framework built on top of Flower

This `README` provides essential information about `slower` from a user's perspective, including how to run the examples provided in the repository.

## Motivation

Flower (`flwr`) is a federated learning (FL) framework that allows practitioners to simulate FL applications using `ray` and deploy them using `gRPC`. One possible limitation of the framework is that clients do not have the option to directly invoke logic to be executed on the server. This functionality is necessary for example if a practitioner needs to train a model using split learning (SL), where some layers are trained on the client and the remaining layers on the server.

Hence enter `slower`, a split learning framework built on top of `flwr`. The framework reuses most of `flwr`'s internal mechanisms and provides the same API. This means that the code written for `flwr` can be used with `slower` without modification.

The main novelty of `slower` is that `Client` objects are seamlessly set an attribute named `server_model_proxy`, which is an interface that allows them to invoke arbitrary code to be executed on `ServerModel` objects - such objects reside on the server. This feature enables users to easily implement SL algorithms, test them in a simulation environment, and deploy them with gRPC.

## Installation

To download `slower` issue the following command:

```bash
python3 -m pip install git+https://git@github.com/sands-lab/slower@master
```

To run the examples in this repository, clone the repository and install the required dependencies as follows:

```bash
git clone https://github.com/BorisRado/slower_simple_examples.git

pip install -r requirements.txt \
    --find-links https://download.pytorch.org/whl/torch_stable.html
```

**Note:** `slower` requires `flwr` version `1.6`. Other versions are not supported. The versions of the other `pip` dependencies in `requirements.txt` are not strict.

## Usage examples

The `examples` folder contains examples on how `slower` can be used to develop the following split learning algorithms:

- `plain`: The client has the lowermost layers, and the server has the uppermost layers of the model. The client performs the forward pass on its model, sends the resulting "smashed embeddings" and target labels to the server, which continues the forward pass, computes the loss, and starts the backward pass. Finally, the server sends the gradient to the client. This usage example is implemented in `examples/plain`. For further information about the algorithm refer to [Distributed learning of deep neural network over multiple agents](https://arxiv.org/pdf/1810.06060)
- `u_shaped`: U-shaped architecture, which extends the plain architecture by having both the lowermost and the uppermost layers. This usage example is implemented in `examples/u_shaped`. For further information about the algorithm refer to [Split learning for health: Distributed deep learning
without sharing raw patient data](https://arxiv.org/pdf/1812.00564).
- `streaming`: SL version, which corresponds to freezing all the client layers. The client is seen as a data (embeddings and target label) producer, while the server is a data consumer. The client computes the embeddings, sends them to the server and immediately continues with processing the next batch, showcasing the streaming version of `slower` (we discuss the streaming capabilities of `slower` below). The server receives the embeddings and target labels, finishes the forward pass, computes the loss, and updates the server-side model. The main difference with the plain SL is that the server does not return any data to the client. This usage example is implemented in `examples/streaming`.

On a high-level, these algorithms may be schematized as follows:

|                              | Client has classification head   | Loss is computed on the server   |
|------------------------------|----------------------------------|----------------------------------|
| Client requires grad         | `u_shaped`                       | `plain`                          |
| Client does not require grad |                                  | `streaming`                      |

### How to run the examples?

To run the examples in the simulation environment (i.e., using ray), run the following (note, that in this case you also need to install `flwr[simulation]==1.6.0`):
```bash
python sl.py configuration=plain
python sl.py configuration=u_shaped
python sl.py configuration=streaming
```

To run the examples by actually having the client and server on different nodes, you can take a look at `simulate_grpc_experiments.sh`, which runs two nodes on a Slurm system. Otherwise, you may run the following on your local machine:

```bash
#!/bin/bash
for configuration in streaming plain u_shaped; do
    python -u run_server.py configuration=$configuration &
    sleep 10  # give some time to the server to start
    python -u run_client.py configuration=$configuration &
    wait
done
```

## Framework overview

### Client

`slower` `Client` objects share the same API as `flwr` `Client`s. This means the user needs to implement a class object with the `fit`, `evaluate`, and possibly `get_parameters` methods. The key difference is that `slower` clients have an attribute named `server_model_proxy` (*note*: the attribute is not available in the `__init__` method), which enables them to invoke logic that is executed by a `ServerModel` object (as of now, each client is associated with its own `ServerModel`). Note that any sampled client can initialize this kind of communication with the server. This, however, does not allow any client to invoke server logic. That is, when sampling clients the server implicitly gives permission to invoke server logic. The permission is revoked at the end of the round.

The client can call any function as long as a method with the corresponding name is defined in the `ServerModel`. For instance:

```python
# Client object
result = self.server_model_proxy.add_arrays(
    a=np.array([1,2,3]),
    b=np.array([2,3,2])
)
```
This will send two `numpy` arrays to the server, and the server will pass them to the `add_arrays` method in the `ServerModel` object associated with the client:

```python
# ServerModel object
def add_arrays(self, a, b):
    return a + b
```

Note that arguments in the method invocation must be provided as keyword arguments, and the same argument names must be used on the server side.

The client call to `add_arrays` is blocking by default, meaning the client will wait until it receives a response from the server. In this case, the client will continue execution once it receives `result = np.array([3, 5, 5])`.

There is also the option to make the calls to the server model in a "streaming" fashion, which means that the client invokes the method and continues with its process. This is done by setting `blocking=False` in the method invocation. For instance:

```python
for v in range(10):
    self.server_model_proxy.add_value(value=np.array([v]), blocking=False)
# `.close_stream()` waits until the server processes all the
# requests and returns the final server's response
response = self.server_model_proxy.close_stream()
```

On the server side:

```python
def add_value(self, value):
    self.sum += value

def get_synchronization_result(self):
    # the value returned by the get_synchronization_result method are
    # returned to the client in the `close_stream()` method
    return self.sum
```

The streaming API works as the name suggest through a stream: the client puts data into a single queue, and the data in the queue is given to the server model sequentially, i.e., the server model needs to complete serving the first request before proceeding to the next one. Note that as of now all the communication happens through a single stream.

**Things to keep in mind**

1. The arguments to the function must be provided with names. For example, you cannot invoke `self.server_model_proxy.some_method(np.array([1, 33, 2]))`; instead, you must use `self.server_model_proxy.some_method(param_name=np.array([1, 33, 2]))`.

2. Any parameter given to the method invocation is given to the corresponding method on the server. The only exceptions are the `blocking` and the `timeout` arguments, which control the internal mechanisms of how the framework routes the request to the server (*note*: the timeout parameter is still not fully integrated).

3. When using the `streaming` API (i.e., setting `blocking=False` in the method invocations), you must call the `close_stream()` method before returning (i.e., in the client's `fit` or `evaluate` method, you must call `close_stream()` before `return ...`). *Note*: if you want the return data to be in the numpy format, use the `numpy_close_stream()` method instead.

4. The arguments of the method must be either all `numpy` arrays and lists of `numpy` arrays (i.e., `Union[np.ndarray, List[np.ndarray]]`), or native `BatchData` objects. For instance:
```python
self.server_model_proxy.method(
    a=[np.array([1,2])], # list of numpy arrays are allowed
    b=np.array([1,2])  # numpy arrays are allowed
)  # is valid
self.server_model_proxy.method(a=2, b=3.2, c="hello")  # built-in data types are not allowed
```
The returned value will be in the same format as the request data (i.e., if the client sends `BatchData` the return will be a `BatchData` object, while if sending `numpy` data the return value will be deserialized to `numpy` data)

5. As `flwr`, `slower` has a raw `Client` and a `NumpyClient` classes (by raw client we mean the `Client` that uses the native data such as `EvaluateIns`, `FitRes`, and so on).

### Server Model

This class is not present in `flwr`. It represents the objects, where the logic invoked by the client is executed. The `ServerModel` consists of the following pre-defined methods:

```python
class NumPyServerModel:

    def get_parameters(self) -> NDArrays:
        # invoked before any training to get the initial server
        # parameters and after every `fit` round concludes to get
        # the updated model parameters

    def configure_fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> None:
        # invoked before the fit round. `parameters` and `config`
        # are equivalent to the corresponding arguments in the client
        # `fit` method

    def configure_evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> None:
        # invoked before the evaluation round. `parameters` and
        # `config` are equivalent to the corresponding arguments
        # in the client `evaluate` method

    def get_synchronization_result(self) -> np.ndarray:
        # get the values that are returned to the `close_stream`
        # method when using the `streaming` API
```

Apart from these methods, the `ServerModel` can have an arbitrary number of methods, that can be invoked by the client (*Note*: the client can call any non-predefined method whose name does not start with an underscore). For instance:

```python
from slower.server.server_model.numpy_server_model import NumPyServerModel


class ServerModelExample(NumPyServerModel):
    ...  # override the predefined methods

    def add_values(self, a, b):
        # can be called by the client
        return a + b

    def _temp(self):
        # the method cannot be called by the client because it starts with "_"
        return None

    def configure_fit(self, parameters, config):
        # cannot be called by the client because it is not a "custom logic" function
        set_parameters(self.model, parameters)
        self.model.train()
        ...
```

As for the client, there is a raw `ServerModel`, which receives custom data types such as `BatchData` and `ServerModelFitIns`, and the `NumPyServerModel`, in which the arguments are deserialized to numpy arrays or lists of numpy arrays.

In the case of `NumPyServerModel`, custom (logic) functions can return one of the following data types:

- dictionaries with `string` keys and `np.ndarray`/`List[np.ndarray]` values.
- single `np.ndarray` values.
- list of `np.ndarray` values.
- `None`: this is only possible for methods invoked with the streaming API.

For instance:
```python
from slower.server.server_model.numpy_server_model import NumPyServerModel


class ServerModelExample(NumPyServerModel):
    ...  # override the predefined methods

    def add_values(self, a, b):
         # returning a single np.ndarray is ok
        return a + b

    def multiple_operations(self, a, b):
        # dictionaries with string keys and numpy values are ok
        return {"sum": a+b, "difference": [a - b, b - a]}

    def get_current_state(self):
        # list of numpy array is ok
        return [self.state_0, self.state_1]

    def add_value(self, a):
        # it is possible to return None only if the method is invoked in a streaming fashion
        self.sum += a
```

### Strategy

A `Strategy` in `slower` server the same purpose as a strategy in `flwr`, i.e., it aggregates client/server parameters, it initializes client/server parameters, it samples clients, and so on. This way, a practitioner can implement the SplitFed algorithm.

In other words, in `flwr` the strategy configures the client parameters for the current round in the `configure_fit` method. A method with the same name and same purpose is present also in the `slower` strategy class. However, in addition to this method, the `slower` strategies also have a `configure_server_fit` method, which configures the parameters that are given to the server models. Similar considerations hold true also for the aggregation methods (`aggregate_fit`, `aggregate_server_fit`). Consequently, if the user implements a weighted average of both the client and server model weights, it effectively obtains an algorithm that is equivalent to FedAvg.

For a start, use the `slower.server.strategy.plain_sl_strategy.PlainSlStrategy`, which does exactly what we just described - i.e., the parameters of both the server models and clients are separately averaged after every training round.


## Limitations and Future work

- As of now, each client is associated with its own `ServerModel`. In the future, an option should be given to allow more flexibility (e.g., let all clients share the same `ServerModel` or let a set of clients share the same `ServerModel`)
- Add native integration with PyTorch, so that users can exchange tensors instead of numpy arrays.
- Integrate the `timeout` parameter to give a maximum time for the server's response.
- Add support for heterogeneous server models (for instance, a powerful client might have a lot of layers hence its server model can be small, while a computationally constrained client might have a shallower model and hence its corresponding server model should compensate by having more layers).


**Final remark**: Suggestions for any update/improvement/feature are always welcome :)
