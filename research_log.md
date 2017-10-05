
2nd October 2017
----------------

Found that an L1 coefficient of 0.0001 with resnet50 is [probably too
high](https://gist.github.com/gngdb/58ad2b20abdfe6c42fa223004747d28a). So
will be using 0.00005. 

In the process of running the non sparse benchmarks with different sizes.

4th October 2017
----------------

Results benchmarking performance as a function of size (with reduced size
experiments) can be found
[here](https://gist.github.com/gngdb/4d70b62addd4a96885e981e837645353).

Sparse results would be today, but there was a bug in the `hyperband.py`
parsing of the l1 argument, so they were all run with an l1 of zero.
Luckily, that means that `main.py` just went over all the non-sparse
experiments again, filling in any that it had missed.

5th October 2017
----------------

Sparse results are _still_ pending. Was running a VGG16 experiment on
another machine and AFS synced the changes when I started the experiment
running sparsity, so ended up running VGG16 when I meant to run an
experiment with resnet50. Will have to be tomorrow.

