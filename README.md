# Slower - Usage examples

This repository contains several examples of how to use `slower` to develop split learning applications.

## Structure

Currently, `slower` supports two dimensions of split learning training:
1. clients can train the lowermost layers (useful when training a model from scratch) or freeze their layers (useful when finetuning a pre-trained model).
2. **architecture**: `slower` supports the plain SL architecture, wherein clients share the embeddings *and* the target labels with the server, and the **U-shaped** architecture, in which clients only share the embeddings and compute the loss themselves.

The way we refer to these dimensions is the following:

|                              | Client has classification head   | Loss is computed on the server   |
|------------------------------|----------------------------------|----------------------------------|
| Client requires grad         | `plain`                          | `u_shaped`                       |
| Client does not require grad |                                  | `streaming`                      |
