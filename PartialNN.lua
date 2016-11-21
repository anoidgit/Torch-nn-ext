local PartialNN, parent = torch.class("nn.PartialNN", "nn.Container")

function PartialNN:__init(module,nForward)
	parent.__init(self)
	self.module = module
	self.modules = {module}
	self.nForward = nForward
end

function PartialNN:updateOutput(input)
	local _ndim = input:nDimension()
	local _isize = input:size()
	local _nstart = self.nForward+1
	local _nsize = _isize[_ndim]-self.nForward
	self.output = input:new():resize(_isize)
	self.output:narrow(_ndim,1,self.nForward):copy(self.module:updateOutput(input:narrow(_ndim,1,self.nForward)))
	self.output:narrow(_ndim,_nstart,_nsize):copy(input:narrow(_ndim,_nstart,_nsize))
	return self.output
end

function PartialNN:updateGradInput(input, gradOutput)
	local _ndim = gradOutput:nDimension()
	local _isize = input:size()
	local _nstart = self.nForward + 1
	local _nsize = _isize[_ndim] - self.nForward
	self.gradInput = gradOutput.new():resize(_isize)
	self.gradInput:narrow(_ndim,1,self.nForward):copy(self.module:updateGradInput(input:narrow(_ndim,1,self.nForward),gradOutput:narrow(_ndim,1,self.nForward)))
	self.gradInput:narrow(_ndim,_nstart,_nsize):copy(gradOutput:narrow(_ndim,_nstart,_nsize))
	return self.gradInput
end

function PartialNN:accGradParameters(input, gradOutput, scale)
	local _ndim = gradOutput:nDimension()
	self.module:accGradParameters(input:narrow(_ndim,1,self.nForward), gradOutput:narrow(_ndim,1,self.nForward), scale)
end

function PartialNN:accUpdateGradParameters(input, gradOutput, lr)
	local _ndim = gradOutput:nDimension()
	self.module:accUpdateGradParameters(input:narrow(_ndim,1,self.nForward), gradOutput:narrow(_ndim,1,self.nForward), lr)
end

function PartialNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
	local _ndim = gradOutput:nDimension()
	self.module:sharedAccUpdateGradParameters(input:narrow(_ndim,1,self.nForward), gradOutput:narrow(_ndim,1,self.nForward), lr)
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
