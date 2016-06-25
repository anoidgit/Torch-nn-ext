require "nn"
require "nngraph"

function getresmodel(modelcap,scale,usegraph)
	local rtm=nn.ConcatTable()
	rtm:add(modelcap)
	if not scale or scale==1 then
		rtm:add(nn.Identity())
	elseif type(scale)=='number' then
		rtm:add(nn.Sequential():add(nn.Identity()):add(nn.MulConstant(scale,true)))
	else
		rtm:add(nn.Sequential():add(nn.Identity()):add(scale))
	end
	local rsmod=nn.Sequential():add(rtm):add(nn.CAddTable())
	if usegraph then
		local input=nn.Identity()()
		local output=rsmod(input)
		return nn.gModule({input},{output})
	else
		return rsmod
	end
end
