require "nn"

function getresmodel(modelcap,scale)
	local rtm=nn.ConcatTable()
	rtm:add(modelcap)
	if not scale then
		rtm:add(nn.Identity())
	else
		rtm:add(nn.Sequential():add(nn.Identity()):add(nn.MulConstant(scale,true)))
	end
	return nn.Sequential():add(rtm):add(nn.CAddTable())
end
