# Torch-nn-ext
some extention to torch and nn package

 * [getmaxout](#getmaxout) : Get a maxout module;
 * [getres](#getres) : Get a residue module;

<a name='getmaxout'></a>
### [maxoutmodule] getmaxout(inputsize,outputsize,nlinear) ###
getmaxout(inputsize,outputsize,nlinear) will return a maxout module, contains `nlinear` parallel linear module(nn.Linear()) with size(`inputsize`,`inputsize`), the inputs and outputs data size should be like batch*datasize.

<a name='getres'></a>
### [residuemodule] getres(module_encap,scale) ###
getres(module_encap,scale) returns a residue module by encapsulate `module_encap`, `scale` controls how much you want to residue, default is 1, keep nil will makes it a little bit efficient. In the most simple use `module_encap` could be just a nn.Linear(), module_encap's inputs size must be the same with the outputs size.
If you encapsulate nn.Tanh() with getres(), you will get a Residue-Activation(res-tanh) Function, which does not saturate, converges much faster, will not “die”, zero mean outputs, computationally efficient, and does not times the number of parameters/neuron like maxout.
