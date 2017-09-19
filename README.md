# Fast Simple Hyperparameter Search with PyTorch

The goal is to make a hyperparameter that is robust by being simple and
runs quickly. Typically this is for comparing two different models, where
you need to be sure one is not better just by luck in stochastic
optimisation.

The trick making this optimisation fast is the extremely fast model
construction and loading times in PyTorch. We can tear down and construct
models in a fraction of the time it takes to train an epoch, which would
not be possible in Tensorflow or Theano. So, on every iteration we're
loading a selection of models from disk, training for an epoch and then
saving back to disk. This means we can compare a variety of models
concurrently, *and* we can feed the same minibatch to all models when they
are training on separate GPUs.

Training in this way is ideally suited to [Hyperband][], but there are
other methods we could also use. The intuition behind this model is that we
are throwing away half of the worst performing models after training them
all for a specified length of time.

[hyperband]: https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

## Checkpoints

The `Checkpoint` object we've defined handles saving, loading, forward and
backward passes through our model. Models are saved using `torch.save` in
dictionaries with the following attributes:

* `'epoch'`: the number of epochs the model has been trained
* `'acc'`: validation accuracy at that epoch
* `'net'`: the network itself

The filename of each checkpoint itself contains the hyperparameter settings
and also contains the epoch and accuracy. These are given to an arbitrary
number of significant digits that is kept constant.

## Checkpoint Handlers

These decide which checkpoints to hand to the epoch iterator. On each
iteration, the `get_next_checkpoint` method is called a number of times to
return a new set of checkpoints. These are then passed to the `train` and
`validate` functions to run the training and validation in parallel on as
many GPUs as are available.

# DICE Setup

Export `SCRATCH` as a environment variable before running to choose where
to save data and checkpoints:

```
export SCRATCH=<somewhere on scratch like /disk/scratch/you>
```

## Accuracy

Should note that the accuracies we're comparing against in this table are
for [models trained
on the full training set][kuangliu], whereas we use a validation split for model
comparison. We could overfit to the test dataset and get higher accuracies.

| Model             | Acc.        | Acc. after optimisation |
| ----------------- | ----------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)             | 92.64%      | ??? |

[kuangliu]: https://github.com/kuangliu/pytorch-cifar
