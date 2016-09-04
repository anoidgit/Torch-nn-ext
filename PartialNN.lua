local PartialNN, parent = torch.class("nn.PartialNN", "nn.Container")

function PartialNN:__init(module,nPartial)
	parent.__init(self)
	self.module = module
	self.nPartial=nPartial
end

function PartialNN:updateOutput(input)
	self.input = input or self.input
	local ndim = self.input:nDimension()
	local nkeep = self.input:size(ndim) - self.nPartial
	self.output = torch.cat(self.module:updateOutput(self.input:narrow(ndim,1,nkeep)),self.input:narrow(ndim,nkeep+1,self.nPartial))
	return self.output
end

function PartialNN:updateGradInput(input, gradOutput)
	self.input = input or self.input
	self.gradOutput = output or self.gradOutput
	local ndim = self.gradOutput:nDimension()
	local nkeep = self.gradOutput:size(ndim) - self.nPartial
	self.gradInput = torch.cat(self.module:updateGradInput(self.gradOutput:narrow(ndim,1,nkeep)),self.gradOutput:narrow(ndim,nkeep+1,self.nPartial))
	return self.gradInput
end

function PartialNN:accGradParameters(input, gradOutput, scale)
	self.input = input or self.input
	self.gradOutput = output or self.gradOutput
	local ndim = self.gradOutput:nDimension()
	local nkeep = self.gradOutput:size(ndim) - self.nPartial
	self.module:accGradParameters(input:narrow(ndim,1,nkeep), gradOutput:narrow(ndim,1,nkeep), scale)
end

function PartialNN:accUpdateGradParameters(input, gradOutput, lr)
	self.input = input or self.input
	self.gradOutput = output or self.gradOutput
	local ndim = self.gradOutput:nDimension()
	local nkeep = self.gradOutput:size(ndim) - self.nPartial
	self.module:accUpdateGradParameters(input:narrow(ndim,1,nkeep), gradOutput:narrow(ndim,1,nkeep), lr)
end

function PartialNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
	self.input = input or self.input
	self.gradOutput = output or self.gradOutput
	local ndim = self.gradOutput:nDimension()
	local nkeep = self.gradOutput:size(ndim) - self.nPartial
	self.module:sharedAccUpdateGradParameters(input:narrow(ndim,1,nkeep), gradOutput:narrow(ndim,1,nkeep), lr)
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
