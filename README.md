# Torch-nn-ext
some extention to torch and nn package

 * [getmaxout](#getmaxout) : Get a maxout module;
 * [getresmodel](#getresmodel) : Get a residue module;
 * [createlineardecaydpnn](#createlineardecaydpnn) : Get a deep neural network, the number of hidden units decay linearly.
 * [vecLookup](#vecLookup) : LookupTable with initialization matrix;
 * [graphmodule](#graphmodule) : generate a module with nngraph, maybe faster;

<a name='getmaxout'></a>
### [maxoutmodule] getmaxout(inputsize,outputsize,nlinear) ###
getmaxout(inputsize,outputsize,nlinear) will return a maxout module, contains `nlinear` parallel linear module(nn.Linear()) with size(`inputsize`,`outputsize`), the inputs and outputs data size should be like batch*datasize.

This function takes 3 arguments:
 * `inputsize` : the size of the input.
 * `outputsize` : the size of the output.
 * `nlinear` : number of the linear you want to parallel.

<a name='getresmodel'></a>
### [residuemodule] getresmodel(module_encap,scale,usegraph) ###
getresmodel(module_encap,scale,usegraph) returns a residue module by encapsulate `module_encap`, `scale` controls how much you want to residue, default is 1, keep nil will makes it a little bit efficient. In the most simple use `module_encap` could be just a nn.Linear(), `module_encap`'s inputs size must be the same with the outputs size, `usegraph` need nngraph and may get a faster module.
If you encapsulate nn.Tanh() with getresmodel(), you will get a Residue-Activation(res-tanh) Function, which does not saturate, converges much faster, will not “die”, zero mean outputs, computationally efficient, and does not times the number of parameters/neuron like maxout.
I have roughly changed the GRU, just replace the tanh with scaled residue tanh, It converges faster and get a better result from Language Model test on PTB, It was not just theory better, but actually better in experiment with deep neural network. The module and the test script can be find at test/.

This function takes 3 arguments:
 * `module_encap` : the module you want to encapsulate with residue connection.
 * `scale` : how much you want to scale the residue connection, default is 1, without any scalar.
 * `usegraph` : use nngraph instead of container to try to get a faster module.

<a name='createlineardecaydpnn'></a>
### [dpnn] createlineardecaydpnn(input,output,step,minfc,transfer) ###
Create a deep neural network easily with this function. It will create a deep neural network with the number of hidden units decay linearly.

This function takes 5 arguments:
 * `input` : the size of the input.
 * `output` : the size of the output.
 * `step` : the output size of the hiddenlayers decay `step` layer by layer.
 * `minfc` : the output size of the hiddenlayers will always larger than `minfc`. when it will decay below `minfc`, the stack process will be stoped, and add a final output layer.
 * `transfer` : a Module that processes the output of the `merge` Module and output a time-step's output of the FastResidueRecurrent Module.

<a name='vecLookup'></a>
### vecLookup ###

The `nn.vecLookup(vecin, paddingValue, maxNorm, normType)` takes 4 arguments:

 * `vecin` : the initialization vectors, it should be a tensor with size (#vocab*#vector).
 * `paddingValue` : same with nn.LookupTable.
 * `maxNorm` : same with nn.LookupTable.
 * `normType` : same with nn.LookupTable.

<a name='graphmodule'></a>
### [graphmodule] graphmodule(module_graph) ###
Reconstruct `module_graph` with nngraph, try make it faster, but it need some time and may make computation less module like LookupTable worse.

This function takes only 1 arguments:
 * `module_graph` : the module to rebuild with nngraph.
