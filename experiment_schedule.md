

# Resnet50

Doing experiments with 64, 32, 16 and 8 planes in first block. The number
of parameters in the network are approximately scaled with the square of
this value.

Hyperband search uses the following settings:

* `eta` controlling the successive halving of 3
* `max_iter` (maximum number of iterations any model can be run) 

Non sparse resnet with default sizing:

* ~~Resnet50 with 8 planes at input (model multiplier of 1)~~
* ~~Resnet50 with 16 planes at input (model multiplier of 2)~~
* ~~Resnet50 with 32 planes at input (model multiplier of 3)~~
* ~~Resnet50 with 64 planes at input (model multiplier of 4)~~

Sparse with L1 coefficient of 0.00005:

* ~~Resnet50 with 8 planes at input (model multiplier of 1)~~
* ~~Resnet50 with 16 planes at input (model multiplier of 2)~~
* ~~Resnet50 with 32 planes at input (model multiplier of 3)~~
* ~~Resnet50 with 64 planes at input (model multiplier of 4)~~

## Replicated Methods

Working with [Deep Compression][dc] and [Sparsifying Variational
Dropout][spvar].

Sparsifying variational dropout with default parameters:

* Resnet50 with 8 planes at input (model multiplier of 1)
* Resnet50 with 16 planes at input (model multiplier of 2)
* Resnet50 with 32 planes at input (model multiplier of 3)
* Resnet50 with 64 planes at input (model multiplier of 4)

Deep compression with ad hoc settings (good default settings unknown, not
published):

* Resnet50 with 8 planes at input (model multiplier of 1)
* Resnet50 with 16 planes at input (model multiplier of 2)
* Resnet50 with 32 planes at input (model multiplier of 3)
* Resnet50 with 64 planes at input (model multiplier of 4)

[dc]: https://arxiv.org/abs/1510.00149v5
[lw]: https://arxiv.org/abs/1506.02626
[spvar]: https://arxiv.org/abs/1701.05369
