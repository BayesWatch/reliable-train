
# Hyperband interface

The command line interface to the `hyperband.py` script is just that the
script implementing your experiment be provided as the first positional
argument:

```
python hyperband.py <your-script>.py
```

There are also some optional arguments to control how hyperband works,
which you can view by:

```
python hyperband.py -h
```

*However*, while this interface is simple, it makes several assumptions
about your script (the first of which is that your script is written in
Python). The following list looks scary, you may prefer for me to just say
it has to have the same functionality as [`dummy.py`](dummy.py).

1. Your script will *only run the experiment when called from the
command line* (another way of saying this is that it correctly incorporates
[`if __name__ == '__main__'`](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)).
2. It has an optional argument allowing us to specify how many iterations
to run, and this is called `--epochs`. This is how many to run **in
total**, not additional; ie if your model has been trained for 10 epochs
and you want to train for 10 more, you would specify `--epochs 20`.
3. The final output printed *on it's own line* is the value you want to
minimise.
4. It implements a function called `get_random_config_id` that takes a
`np.random.RandomState` as its argument, and produces a string that fully
describes a particular randomly sampled configuration. For example, it
could produce `lr:0.01.batch_size:256`, because we can recover those
hyperparameter settings from this string using...
6. Your script must implement a function called `run_identity` that takes
any optional arguments your script could receive, and returns a string
uniquely defining these. This allows your script to implement more than one
possible hyperband experiment. For example, if you would like to run a
separate hyperparameter search with hyperband on different models, you
could make an optional argument that would do this. Then, your
`run_identity` function just needs to find the `--model` argument and
return it as a string.
7. It takes `config_id` as the first positional argument on the command
line.
8. It saves the model it's currently training after training for a given
number of epochs and this model is saved to a file that is unique *for that
configuration setting*.
9. It loads models from a pre-existing checkpoint when it can.
10. Your script must be able to allocate itself to a given GPU, and this
must be accessible by an optional argument `--gpu`. In addition, it should
either be able to run on multiple GPUs with the `--multi_gpu` option, or it
should be able to ignore that option (for example by using
[parse_known_args](https://docs.python.org/3.4/library/argparse.html#argparse.ArgumentParser.parse_known_args).

The `hyperband.py` script will also catch any errors your script throws and
write them to its log file. It will also assign an arbitrary high loss
value of 100 whenever this happens. If you are doing some experiment where
100 isn't an arbitrarily high loss then this could cause problems.

This may seem like a lot of requirements, but there's not much that can be
done about it. This is not a black box optimiser, so it has to make some
assumptions about the box being optimised. For a concrete example, see the
script `dummy.py` implementing a dummy experiment for optimisation. 

