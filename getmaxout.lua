require "nn"

function getmaxout(inputs,outputs,nlinear)
	local rtm=nn.ConcatTable()
	local ncyc=nlinear or 2
	for i=1,ncyc do
		local tmpli=nn.Linear(inputs,outputs)
		rtm:add(tmpli:clone())
	end
	return nn.Sequential():add(rtm):add(nn.JoinTable(2)):add(nn.Reshape(ncyc,outputs,true)):add(nn.Max(2))
end
