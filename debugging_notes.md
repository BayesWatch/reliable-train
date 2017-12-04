
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

1st Decembet 2017
-----------------

Managed to solve the problem with subprocess by killing the whole process
group. Started using `CUDA_VISIBLE_DEVICES` to solve the second problem,
which should solve it.
