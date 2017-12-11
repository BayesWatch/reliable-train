
28th November 2017
------------------

Problems with subprocess:

1. When using `process.terminate()` process disappears from list of
processes on GPU, but GPU memory still full. Also, causes the program to
hang waiting at the next `communicate()`

Have also noticed that the `satisficing.py` script is not managing to keep
everything on one GPU. Every process that has stuff on the other GPUs
is *also* allocating memory on GPU 0. I have no idea why this is happening,
because the only times `.cuda` is called, the GPU index is supplied, and
it's global to the script. May be necessary to using
`CUDA_VISIBLE_DEVICES`.

1st December 2017
-----------------

Managed to solve the problem with subprocess by killing the whole process
group. Started using `CUDA_VISIBLE_DEVICES` to solve the second problem,
which should solve it.

11th December 2017
------------------

Debugging the script meant to run experiments on Bayesian Compression. The
script will run, and is running with VGG16. However, initial experiments
with ResNet50 (it was a mistake to run this first) didn't converge. I
suspect this is something to do with the weight of regularisation; the ELBO
is *huge*. Sample input and output during training:

```
> python bayes.py 0.1_0.1_64 --model VGG16 -v
Step: 168ms | Tot: 2m8s | | 0 | 188771.970 | 15.239% | 0.920 | 
```

I know that VGG16 has a lot more parameters in it than the experiments in
the Bayesian Compression Tutorial, but that still seems a little high, and
I can see why it might not converge in these conditions.

To be sure, the best thing for me to do is probably to run the [tutorial
code](https://github.com/BayesWatch/Tutorial_BayesianCompressionForDL) and
see what the losses it reports during training are.

*Done that*: looks like the ELBO does grow with the number of parameters in
your model, as you might expect. But, it's still far lower for their
examples. Also, there is no convolution architecture in the tutorial to get
a better comparison against. Inconclusive.

After training VGG16 to near the end of the first epoch, training produces
NaNs, and these propagate throughout the system.

This doesn't happen if we disable the variational loss, so something's
going wrong.

Easiest way to debug this is going to be to change the model to a simple
MLP and see if this can converge as in the tutorial example; CIFAR-10 is
harder than MNIST, but a simple MLP still shouldn't produce NaNs.

Wrote a simple MLP learning from the channel-wise mean pixels, and
confirmed that without the kl divergence loss it does converge.

After including the kl divergence loss, the elbo is about 200 times larger
than in the tutorial. Suspect my function that is supposed to gather kl
divergences from the layers composing a network is quite broken.

Checked by substituting the example method (built-in method in the Net
class), and got the same error.

Appears that it is probably that I'm using a broken way to get the size of
the training set `len(trainloader)` returns the number of minibatches,
*not* the number of examples. Fixing that fixes my MLP, but may not fix
VGG16. Also, we still have no assurance that we're always going to be able
to avoid NaNs.

After running for 33 epochs (getting to 85% accuracy on validation) the
VGG16 Bayesian Compression model starts to hit NaNs.

Solution to this is probably to use the `clip_variances` function in the
example. So, putting in a function to implement this.

*Sidenote*: the sparsity this method causes isn't obvious from the
perspective of the sparsity measuring function. We kind of have to know
what the sparsity might be, so we need to implement something new to track
sparsity in just these experiments. May be worth using the functions
already developed in the Bayesian Compression Tutorial.
