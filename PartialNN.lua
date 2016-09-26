local PartialNN, parent = torch.class("nn.PartialNN", "nn.Container")

function PartialNN:__init(module,nForward)
	parent.__init(self)
	self.module = module
	self.nForward=nForward
end

function PartialNN:updateOutput(input)
	local ndim = input:nDimension()
	self.output = input:clone()
	self.output:narrow(ndim,1,self.nForward):copy(self.module:updateOutput(input:narrow(ndim,1,self.nForward)))
	return self.output
end

function PartialNN:updateGradInput(input, gradOutput)
	local ndim = gradOutput:nDimension()
	self.gradInput = gradOutput:clone()
	self.gradInput:narrow(ndim,1,self.nForward):copy(self.module:updateGradInput(input:narrow(ndim,1,self.nForward),gradOutput:narrow(ndim,1,self.nForward)))
	return self.gradInput
end

function PartialNN:accGradParameters(input, gradOutput, scale)
	local ndim = gradOutput:nDimension()
	self.module:accGradParameters(input:narrow(ndim,1,self.nForward), gradOutput:narrow(ndim,1,self.nForward), scale)
end

function PartialNN:accUpdateGradParameters(input, gradOutput, lr)
	local ndim = gradOutput:nDimension()
	self.module:accUpdateGradParameters(input:narrow(ndim,1,self.nForward), gradOutput:narrow(ndim,1,self.nForward), lr)
end

function PartialNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
	local ndim = gradOutput:nDimension()
	self.module:sharedAccUpdateGradParameters(input:narrow(ndim,1,self.nForward), gradOutput:narrow(ndim,1,self.nForward), lr)
end

function PartialNN:__tostring__()
	if self.module.__tostring__ then
		return torch.type(self) .. ' @ ' .. self.module:__tostring__()
	else
		return torch.type(self) .. ' @ ' .. torch.type(self.module)
	end
end

-- useful for multiple-inheritance
function PartialNN.decorate(class)
	class.updateOutput = nn.PartialNN.updateOutput
	class.updateGradInput = nn.PartialNN.updateGradInput
	class.accGradParameters = nn.PartialNN.accGradParameters
	class.accUpdateGradParameters = nn.PartialNN.accUpdateGradParameters
	class.sharedAccUpdateGradParameters = nn.PartialNN.sharedAccUpdateGradParameters
	class.__tostring__ =  nn.PartialNN.__tostring__
end
