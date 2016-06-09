require "nn"

function createlineardecaydpnn(input,output,step,minfc,transfer)
	local ins=input
	local ous=ins-step
	rs=nn.Sequential()
	while ous>minfc do
		rs:add(nn.Linear(ins,ous))
		rs:add(transfer)
		ins=ous
		ous=ins-step
	rs:add(nn.Linear(ous,output))
	return rs
end
