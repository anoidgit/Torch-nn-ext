require "nngraph"

require "cudnn"

function buildrtcnn_simple(vsize)
	local input = nn.Identity()()
	local ws = cudnn.GRU(vsize, vsize, 1)(input)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local output = nn.Select(1, -1)(cudnn.GRU()(newseq))
	return nn.gModule({input}, {output})
end

function buildrcnn_simple(vsize, stdbuild)
	local input = nn.Identity()()
	local ws = cudnn.GRU(vsize, vsize, 1)(input)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local zv = cudnn.GRU(vsize, vsize, 1)(newseq)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.SoftMax()(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.SoftMax()(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Sum(1)(cseq)
	return nn.gModule({input}, {output})
end

function buildrcnn_gatedsimple(vsize, stdbuild)
	local input = nn.Identity()()
	local ws = cudnn.GRU(vsize, vsize, 1)(input)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local zv = cudnn.GRU(vsize, vsize, 1)(newseq)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.SoftMax()(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.SoftMax()(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Select(1, -1)(cudnn.GRU()(cseq))
	return nn.gModule({input}, {output})
end

function buildrcnn_base(vsize, stdbuild)
	local input = nn.Identity()()
	local rgate = cudnn.GRU(vsize, vsize, 1)(input)-- better to find a way which can get bi-directional information
	local nrgate = nn.Sigmoid()(rgate)
	local appr = nn.CMulTable()({input, nrgate})
	local ws = cudnn.GRU(vsize, vsize, 1)(appr)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local zv = cudnn.GRU(vsize, vsize, 1)(newseq)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.SoftMax()(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.SoftMax()(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Sum(1)(cseq)
	return nn.gModule({input}, {output})
end

require "rnn"
function buildrcnn_br(vsize, stdbuild)
	local input = nn.Identity()()
	local rinput = nn.SeqReverseSequence(1)(input)
	local rinfo = cudnn.GRU(vsize, vsize, 1)(rinput)
	local info = nn.SeqReverseSequence(1)(rinfo)
	local binput = nn.ConcatTable(3)({input, info})
	local rgate = cudnn.GRU(vsize * 2, vsize, 1)(binput)
	local nrgate = nn.Sigmoid()(rgate)
	local appr = nn.CMulTable()({input, nrgate})
	local ws = cudnn.GRU(vsize, vsize, 1)(appr)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local zv = cudnn.GRU(vsize, vsize, 1)(newseq)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.SoftMax()(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.SoftMax()(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Sum(1)(cseq)
	return nn.gModule({input}, {output})
end

function buildrcnn_brbu(vsize, stdbuild)
	local input = nn.Identity()()
	local rinput = nn.SeqReverseSequence(1)(input)
	local rinfo = cudnn.GRU(vsize, vsize, 1)(rinput)
	local info = nn.SeqReverseSequence(1)(rinfo)
	local binput = nn.ConcatTable(3)({input, info})
	local rgate = cudnn.GRU(vsize * 2, vsize, 1)(binput)
	local nrgate = nn.Sigmoid()(rgate)
	local appr = nn.CMulTable()({input, nrgate})
	local ws = cudnn.GRU(vsize, vsize, 1)(appr)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.ConcatTable()({w_hat, input})
	local rnewseq = nn.SeqReverseSequence(1)(newseq)
	local rninfo = cudnn.GRU(vsize, vsize, 1)(rnewseq)
	local ninfo = nn.SeqReverseSequence(1)(rninfo)
	local bninput = nn.ConcatTable(3)({ninfo, newseq})
	local zv = cudnn.GRU(vsize * 2, vsize, 1)(bninput)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.SoftMax()(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.SoftMax()(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Sum(1)(cseq)
	return nn.gModule({input}, {output})
end
