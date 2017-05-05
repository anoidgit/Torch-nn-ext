# Torch-nn-ext
some extention to torch and nn package

 * [getmaxout](#getmaxout) : Get a maxout module;
 * [getresmodel](#getresmodel) : Get a residue module;
 * [createlineardecaydpnn](#createlineardecaydpnn) : Get a deep neural network, the number of hidden units decay linearly.
 * [vecLookup](#vecLookup) : LookupTable with initialization matrix;
 * [maskZerovecLookup](#maskZerovecLookup) : like vecLookup but allows to index 0 which results tensors of 0;
 * [graphmodule](#graphmodule) : generate a module with nngraph;
 * [PartialNN](#PartialNN) : decorater a module to a part forward module;
 * [SwapTable](#SwapTable) : exchange inner table and outer table({{1,2,3},{4,5,6}}<=>{{1,4},{2,5},{3,6}});
 * [getgcnn](#getgcnn) : Get a Gated Combination Neural Network;
 * [getrcnn](#getrcnn) : Get a Recurrent Combination Neural Network;
 * [SeqDropout](#nn.SeqDropout) : Dropout for SeqLSTM, SeqGRU and SeqBRNN;

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

The `nn.vecLookup(vecin, dontupdatevec, paddingValue, maxNorm, normType)` takes 5 arguments:

 * `vecin` : the initialization vectors, it should be a tensor with size (#vocab*#vector).
 * `dontupdatevec` : whether update vectors during training.
 * `paddingValue` : same with nn.LookupTable.
 * `maxNorm` : same with nn.LookupTable.
 * `normType` : same with nn.LookupTable.

<a name='maskZerovecLookup'></a>
### maskZerovecLookup ###

The `nn.maskZerovecLookup(vecin, dontupdatevec, paddingValue, maxNorm, normType)` takes 5 arguments:

 * `vecin` : the initialization vectors, it should be a tensor with size (#vocab*#vector).
 * `dontupdatevec` : whether update vectors during training.
 * `paddingValue` : same with nn.LookupTable.
 * `maxNorm` : same with nn.LookupTable.
 * `normType` : same with nn.LookupTable.

<a name='graphmodule'></a>
### [graphmodule] graphmodule(module_graph) ###
Reconstruct `module_graph` with nngraph, try make it faster, but it need some time and may make computation less module like LookupTable worse.

This function takes only 1 arguments:
 * `module_graph` : the module to rebuild with nngraph.

<a name='PartialNN'></a>
### PartialNN ###

The `nn.PartialNN(module, nForward)` was expected to decorate the activation functions whos input size and output size was same, and get a cross-layer neural network, It takes 2 arguments:

 * `module` : the module process the input data.
 * `nForward` : size of the input which was processed by `module`.

<a name='SwapTable'></a>
### SwapTable ###

The `nn.SwapTable()` was expected to exchange inner table and outer table, for example, transform {{1,2,3},{4,5,6}} with {{1,4},{2,5},{3,6}}, It takes no arguments.

<a name='getgcnn'></a>
### [gcnn] getgcnn(ncombine,vecsize,inputdim,usegraph) ###
Create a Gated Combination Neural Network with this function. It need nngraph.

This function takes 4 arguments:
 * `ncombine` : the number of input will be combined into the result embedding.
 * `vecsize` : the size of the input and output vector.
 * `inputdim` : the size of the input data.
 * `usegraph` : whether use nngraph implementation, if `true`, need `nngraph`(https://github.com/torch/nngraph).

<a name='getrcnn'></a>
### [rcnn] getrcnn(vecsize,stdbuild) ###
Create a Recurrent Combination Neural Network with this function. It need nngraph.

This function takes 2 arguments:
 * `vecsize` : the size of the input and output vector.
 * `stdbuild` : apply softmax on the first dimension or not.
 
<a name='nn.SeqDropout'></a>
### SeqDropout ###

A Dropout Module changed from nn.Dropout(https://github.com/torch/nn), used for SeqLSTM, SeqGRU and SeqBRNN of rnn(https://github.com/Element-Research/rnn), the whole sequence will share the same mask which is different from nn.Dropout. The input of this module should be ```seqlen x batchsize x inputsize```.
