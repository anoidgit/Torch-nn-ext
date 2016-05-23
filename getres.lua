require "nn"

function getres(modelcap)
	local rtm=nn.ConcatTable()
	rtm:add(modelcap)
	rtm:add(nn.Identity())
	return nn.Sequential():add(rtm):add(nn.CAddTable())
end
