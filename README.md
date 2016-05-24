# Torch-nn-ext
some extention to torch and nn package

getmaxout(inputs,outputs) returns a maxout module,with size(inputs,outputs),the inputs and outputs data size should be like batch*datasize

getres(module_encap) returns a residue module by encapsulate module_encap, in the most simple use module_encap could be just a nn.Linear, module_encap's inputs size must be the same with the outputs size. If you encapsulate nn.Tanh() with getres(), you will get a res-transfer(res-tanh) function, which does not saturate, converges much faster, will not “die”, zero mean outputs, computationally efficient, and does not times the number of parameters/neuron like maxout.
