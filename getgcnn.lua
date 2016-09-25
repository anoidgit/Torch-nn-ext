function getgcnn(ncomb,vecsize,inputdim)
	local input=nn.Identity()():annotate{name="input",description="input"}
	local rgate=nn.Sigmoid()(nn.Linear(ncomb*vecsize,ncomb*vecsize)(input)):annotate{name="reset gate",description="Get reset gate"}
	local wcommonp=nn.CMulTable()({rgate,input}):annotate{name="reset char",description="use reset gate to get reset char"}
	local wcommon=nn.Tanh()(nn.Linear(ncomb*vecsize,vecsize)(wcommonp)):annotate{name="common vector",description="Common environment vector"}
	local wc=nn.JoinTable(2,2)({wcommon,input}):annotate{name="env&char vector",description="Common environment vector and char vector"}
	local ugate=nn.SoftMax()(nn.Linear((ncomb+1)*vecsize,(ncomb+1)*vecsize)(wc)):annotate{name="update gate",description="update gates"}
	local www=nn.CMulTable()({ugate,wc}):annotate{name="www",description="apply update gates"}
	local wrs=nn.CAddTable()(nn.SplitTable(3,3)(nn.Reshape(vecsize,(ncomb+1),true)(www))):annotate{name="word vector",description="get word vector"}
	return nn.Sequential():add(nn.JoinTable(inputdim,inputdim)):add(nn.Bottle(nn.gModule({input},{wrs})))
end
