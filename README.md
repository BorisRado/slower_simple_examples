# Slower - Split Learning Framework built on top of Flower

This `README` provides essential information about `slwr` from a user's perspective, including how to run the examples provided in the repository.

## Motivation

Flower (`flwr`) is a federated learning (FL) framework that allows practitioners to simulate FL applications using `ray` and deploy them using `gRPC`.
One possible limitation of the framework is that clients do not have the option to directly invoke logic to be executed on the server.
This functionality is necessary for example if a practitioner needs to train a model using split learning (SL), where some layers are trained on the client and the remaining layers on the server.

Hence enter `slwr`, a split learning framework built on top of `flwr`.
The framework reuses most of `flwr`'s internal mechanisms and provides the same API.
This means that the code written for `flwr` can be used with `slower` without modification.

The main novelty of `slwr` is that `Client` objects are seamlessly set an attribute named `server_model_proxy`, which is an interface that allows them to invoke arbitrary code to be executed on `ServerModel` objects - such objects reside on the server.
This feature enables users to easily implement SL algorithms and deploy them on a real-world distributed environment.

## Installation

To download `slwr` issue the following command:

```bash
python3 -m pip install git+https://git@github.com/sands-lab/slower@master
```

To run the examples in this repository, clone the repository and install the required dependencies as follows:

```bash
git clone https://github.com/BorisRado/slower_simple_examples.git

pip install -r requirements.txt \
    --find-links https://download.pytorch.org/whl/torch_stable.html
```

**Note:**
`slwr` requires `flwr` version `1.9.0`.
Other versions are not supported.
The versions of the other `pip` dependencies in `requirements.txt` are not strict.

## Usage examples

The `examples` folder contains examples on how `slwr` can be used to develop the following split learning algorithms:

- *Plain SL*:
The client has the lowermost layers, and the server has the uppermost layers of the model.
The client performs the forward pass on its model, sends the resulting "smashed embeddings" and target labels to the server, which continues the forward pass, computes the loss, and starts the backward pass.
Finally, the server sends the gradient to the client.
This usage example is implemented in `examples/plain`.
For further information about the algorithm refer to [Distributed learning of deep neural network over multiple agents](https://arxiv.org/pdf/1810.06060)
- *U-shaped SL*:
U-shaped architecture, which extends the plain architecture by having both the lowermost and the uppermost layers.
This usage example is implemented in `examples/u_shaped`.
For further information about the algorithm refer to [Split learning for health: Distributed deep learning
without sharing raw patient data](https://arxiv.org/pdf/1812.00564).
- *Streaming SL*:
SL version, which corresponds to freezing all the client layers.
The client is seen as a data (embeddings and target label) producer, while the server is a data consumer.
The client computes the embeddings, sends them to the server and immediately continues with processing the next batch, showcasing the streaming version of `slwr` (we discuss the streaming capabilities of `slwr` below).
The server receives the embeddings and target labels, finishes the forward pass, computes the loss, and updates the server-side model.
The main difference with the plain SL is that the server does not return any data to the client, hence the client can continue processing without waiting for the server.
This usage example is implemented in `examples/streaming`.
This code can easily be exteded with having a classification head on the client and thus implement algorithms such as [FSL](http://comstolab.kaist.ac.kr/library/papers/ICML_Workshop_Split.pdf) and [AdaSplit](https://arxiv.org/abs/2112.01637).
- *Combining client & server loss*:
this project simulates the fact that the client may perform work while waiting for the server's response.
When the client has completed its work, it can request whether the server's response has been received and wait until it is.
To showcase one possible application of this, in this part we train a client with both the encoder and classification head.
The client's head is trained with a client loss, while the encoder is updated with a combination of the client and server loss.
The client computes the forward pass on its classification head while waiting for the server response.


### How to run the examples?

You can take a look at `simulate_grpc_experiments.sh`, which runs two nodes on a Slurm system.
Otherwise, you may run the following on your local machine:

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

As in `flwr`, also in `slwr` we rely on the concept of "server round". By server training round we indicate a training iteration, in which a set of clients is sampled for training, they train and upon completing training, they return the updated client-model weights to the server. The server (specifically, the `Strategy`) then aggregates the client weights to obtain the new version of the client model. Similarly, as in SL in a training round we may train more than one model on the server, also server-side models need to be aggregated at the end of a training round. Note, that this formulation is very flexible: for instance, in some algorithms the client is not supposed (or required) to reveal any weights to the server. In such a case, the client can store to its disk its weights at the end of a training round and load them again in the next training/evaluation round (clients are stateless, so in this case weights need to be loaded again at the beginning of every training/evaluation round).

From a programming perspective, the framework consists of three main components:

* `Client`: locally train a model possibly by leveraging the server to offload part of the computational burden. `slwr` clients are equivalent to their counterparts in `flwr`.
* `ServerModel` receive requests sent by the clients, process such requests and possibly compute some value, that is returned to the client that made the request.
* `Strategy` coordinate the training process.
Notably, strategies determine which clients should train in the current training epoch, configures the clients, configures one or more `ServerModels`, and routes client requests to the server model that should handle them.


We now provide more details about each of these component.

### Client

`slwr` `Client` objects share the same API as `flwr` `Client`s.
This means the user needs to implement a class object with the `fit`, `evaluate`, and possibly `get_parameters` methods.
The key difference is that `slwr` clients have an attribute named `server_model_proxy` (*note*: the attribute is not available in the `__init__` method), which enables them to invoke arbitrary logic that is executed by a `ServerModel` object.
Note that any *sampled* client can initialize this kind of communication with the server.
This, however, does not allow any client to invoke server logic.
That is, when sampling clients the server implicitly gives permission to invoke server logic.
The permission is revoked at the end of the round.

The client can call any method of the server model as long as a method with the corresponding name is defined in the `ServerModel`.
For instance:

```python
# Client object
result = self.server_model_proxy.add_arrays(
    a=np.array([1,2,3]),
    b=np.array([2,3,2])
) # client waits until it receives the response from the server
```
This will send two `numpy` arrays to the server, and the server will pass them to the `add_arrays` method in some `ServerModel`:

```python
# ServerModel object
def add_arrays(self, a: List[np.ndarray], b: List[np.ndarray]):
    return [a[0] + b[0],]
```

The server model receives `typing.List` arguments.
This is because the framework supports grouping multiple client requests into one single request received by the server model.
Consider for instance having a set of clients, and we want to compute the sum of some client's value.
We can have:
```python
# multiple clients
result = self.server_model_proxy.global_sum(value=np.array(self.value))
```
and on the server:
```python
# server
def global_sum(self, value):
    res = sum(value)
    return [res for _ in range(len(value))]
```
Note, that as the server model receives a list of values (one for every client request), it must return a list of values.

If a method of the server model always processes client requests for only one client at a time, you can use the `single_client_format` decorator (`from slwr.server.server_model.utils import single_client_format`) to extract the first element in the argument list. Thus, in the above case, the code on the server would be modified as follows:

```python
# ServerModel object
@single_client_format
def add_arrays(self, a: np.ndarray, b: np.ndarray):
    return a + b
```

#### Invoking logic on the server

Client's calls to server methods are blocking by default, meaning the client will wait until it receives a response from the server.
Other requests types are supported as well:

* *Streaming*: means that the client makes a request and does not expect to receive a response from the server.
This is done by adding a `_type_=RequestType.STREAM,` argument to the method invokation (import the `RequestType` with `from slwr.common import RequestType`).
The corresponding method on the server *must* return `None`.
A simple example of this kind of requests is the following
    ```python
    for v in range(10):
        self.server_model_proxy.add_value(
            value=np.array([v]),
            _type_=RequestType.STREAM
        )
        # the client continues the execution without waiting
    response = self.server_model_proxy.get_final_sum()
    ```
    On the server side:
    ```python
    def add_value(self, value):
        # we assume that len(value) == 1, so that we don't group client requests
        self.sum += value[0]
        # note that the return value is None

    def get_final_sum(self):
        return self.sum
    ```
* *Futures*: means that the client requires a response, however, it does not require the response immediately.
For instance, the client makes a request, continues with some local work, and finally reads the server response.
Currently, only one request of this kind is supported at a time:
    ```python
    future = self.server_model_proxy.some_method(_type_=RequestType.FUTURE)
    # ... perform some local work ...
    response = future.get_response()
    ```

**Things to keep in mind**

1. Arguments must be provided with names.
For example, you cannot invoke `self.server_model_proxy.some_method(np.array([1, 33, 2]))`;
instead, you must use `self.server_model_proxy.some_method(param_name=np.array([1, 33, 2]))`.

2. Any parameter given to the method invocation is given to the corresponding method on the server.
The only exceptions are the `_type_`, `_timeout_`, and `_streams_` arguments, which control the internal mechanisms of how the framework routes the request to the server.
    * `_type_: RequestType`: as explained, this parameter controls the type of request, either blocking, streaming, or with futures.
    * `_timeout_: Optional[float]`: controls the maximum time allowed for a request. This parameter is not yet integrated.
    * `_streams_: bool`: controls whether the framework uses gRPC's unary or streaming requests. By default, streaming requests are used.

3. You have to tell the `server_model_proxy` attribute the type of data to be communicated.
Therefore, at the beginning of every `fit` and `evaluate` method, you have to use one of the following:
    * `self.server_model_proxy.numpy()`: informs the `server_model_proxy` that the client will be sending numpy arrays;
    * `self.server_model_proxy.torch()`: informs the `server_model_proxy` that the client will be sending pytorch tensors;

    If you do not tell the server model proxy the type of data to be communicated, it will expect to receive the native `BatchData` objects.

4. As `flwr`, `slower` has a raw `Client` and a `NumpyClient` classes (by raw client we mean the `Client` that uses the native data such as `EvaluateIns`, `FitRes`, and so on).

### Server Model

This class is not present in `flwr`.
It represents the objects, where the logic invoked by the client is executed.
The `ServerModel` consists of the following pre-defined methods:

```python
class NumPyServerModel:

    def get_parameters(self) -> NDArrays:
        # invoked before any training to get the initial server parameters

    def get_fit_result(self) -> ServerModelFitRes:
        # invoked after a training epoch to get the updated server parameters
        # possibly with additional configuration to be used for aggregating
        # the weights (e.g., number of batches the model has processed).

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
```

Apart from these methods, the `ServerModel` can have an arbitrary number of methods, that can be invoked by the client.
Specifically, **the client can call any non-predefined method whose name does not start with an underscore**.
For instance:

```python
from slower.server.server_model.numpy_server_model import NumPyServerModel


class ServerModelExample(NumPyServerModel):
    ...  # override the predefined methods

    @single_client_format
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

As for the client, there is a raw `ServerModel`, which receives custom data types such as `BatchData` and `ServerModelFitIns`, and the `NumPyServerModel`, in which the arguments are numpy arrays or lists of numpy arrays, thus the length of all such lists is equal. For instance, suppose the server model is given data from two clients to be processed. In such a case, all arguments will have a length of $2$, and all the first elements in the lists will correspond to the data sent by one client, and the second element the data sent by the second client. As output, the method must return a list, with the values to be sent to the corresponding client. There is one exception to this rule, namely if the method takes no arguments. In such a case, the method must return a single value, which is sent to all the clients that invoked the method.

For the `NumpyServerModel` every element in the (input/output) list can be one of the following data types:

- dictionaries with `string` keys and `np.ndarray`/`List[np.ndarray]` values.
- single `np.ndarray` values.
- list of `np.ndarray` values.
- `None`: this is only possible for methods invoked with the streaming API.

For instance:
```python
from slower.server.server_model.numpy_server_model import NumPyServerModel


class ServerModelExample(NumPyServerModel):
    ...
    # in this example we assume that the length of arguments is always 1
    # (i.e., the server model processes one client request at a time)

    def add_values(self, a, b):
         # returning a single np.ndarray is ok
        return [a[0] + b[0],]

    def multiple_operations(self, a, b):
        # dictionaries with string keys and numpy values are ok
        return [{
            "sum": a[0]+b[0],
            "difference": [a[0] - b[0], b[0] - a[0]]
        },]

    def get_current_state(self):
        # list of numpy array is ok
        return [[self.state_0, self.state_1],]

    def add_value(self, a):
        # it is possible to return None only if the method is invoked in a streaming fashion
        self.sum += a[0]

    def exception(self):
        # if the method takes no argument, it must return a single
        # value. The same value is returned to every client
        return np.array([1,])

    def mistake(self, a):
        return 1 # cannot return an integer
```

If using PyTorch, you can use the `@pytorch_format` decorator in front of a logic function to covert the arguments to pytorch tensors.
This decorator converts every numpy array to a pytorch tensor, concatenates them along the first dimension (the batch dimension), and splits the tensor returned by the function into single client responses.
Consider for instance having two clients sending a batch of data with size 16:
```python
gradient = self.server_model_proxy.serve_gradient_request(
    embeddings=embedding, # embeddings has shape [16, ...]
    labels=labels, # labels has shape [16,]
)
```
By default, the `serve_gradient_request` method of the `NumpyServerModel`  will receive two arguments (`embeddings` and `labels`), and each one will contain a list of two numpy arrays.
The `pytorch_format` decorator concatenates these two batches into one single batch, and splits the returned value in such a way, that the slice of $16$ elements of the returned tensor is given to the first client, and the remaining $16$ to the second client. Therefore, you can write:
```python
@pytorch_format
def serve_grad_request(self, embeddings, labels):
    # both embeddings and labels contain two client batches
    # that have been concatenated along the first dimension
    embeddings.requires_grad_(True)
    preds = self.model(embeddings)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    self.optimizer.step()
    return embeddings.grad # the gradient will be split by the decorator into two
```

### Strategy

A `Strategy` in `slwr` serves the same purpose as a strategy in `flwr`, i.e., it aggregates client/server parameters, it initializes client/server parameters, it samples clients, and so on.
In addition to this, the `Strategy` is also responsible to route client requests to an appropriate server model.

#### Strategy configuration

To customize the behavior of the server model you have to use the `configure_server_<fit/evaluate>`.
These methods return list of `ServerModel<Fit/Evaluate>Ins`.
Each such object represents the configuration used to instantiate a server model (thus, the length of the returned list determines the number of models running on the server), and each server model is associated with as `sid` (server model ID).

At the end of training, the Strategy also needs to aggregate the updated weights of the server models (in the same way as `flwr` strategies need to aggregate the weights sent by clients).

**WORK IN PROGRESS:** Apart from managing the server models, the strategy is also responsible to route client requests to server models.
This is achieved by implementing the `cid_to_sid` method, in which the strategy receives the current request status, the method that the client requested to be invoked, and the client id (`cid`) of the client making the request.
The server returns the `sid` of the server model that should handle this request as well as a boolean value indicating whether the current set of request should be given to the server model or whether we want to wait until receiving more client client requests.

For a start, use (and possibly check out if you want to know more about how `slwr` strategies work) the `slower.server.strategy.plain_sl_strategy.PlainSlStrategy`.
If you want to use this strategy, you can configure it as follows:

* `common_server_model=True, process_clients_as_batch=True`: when training, there is only one model on the server.
The server will wait until receiving a request from every client before passing the list of requests to the server model.
* `common_server_model=True, process_clients_as_batch=False`: when training, there is only one model on the server.
As soon as a request is received by a client, it is passed to the server model, i.e., the server model always receives the request of only one client.
* `common_server_model=False, process_clients_as_batch=False`: when training, each client is associated with its own server model.


## Limitations

* Integrate the `timeout` parameter to give a maximum time for the server's response.
* The framework does no longer support running simulations.
This is because the new functionalities of grouping client requests would require possibly all clients to be active at the same time, which is challanging to parallelize optimally.
* On the server the GIL is a bottleneck, which prevents us from achieving top performances.
We believe the framework is great for algorithm prototyping, however, for deployment, one would have to consider having a framework in a different, better performing language.
This is currently under consideration.

## Possible improvements

* It would be nice to add automatic backward gradient computation.
* Improve the routing mechanism.

**Final remark**: Suggestions for any update/improvement/feature are always welcome :)
