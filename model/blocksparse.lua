------------------------------------------------------------------------
--[[ BlockSparse ]]--
-- A 3 layer model of Distributed Conditional Computation
------------------------------------------------------------------------
local BlockSparse, parent = torch.class("dp.BlockSparse", "dp.Layer")
BlockSparse.isBlockSparse = true

function BlockSparse:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, n_block, hidden_size, window_size, gater_size, 
      output_size, noise_std, threshold_lr, alpha_range, sparse_init, typename
      = xlua.unpack(
      {config},
      'BlockSparse', 
      'Deep mixture of experts model. It is three parametrized '..
      'layers of gated experts. There are two gaters, one for each '..
      'layer of hidden neurons between nn.BlockSparses.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='n_block', type='table', req=true,
       help='number blocks in hidden layers between nn.BlockSparses.'},
      {arg='hidden_size', type='table', req=true,
       help='number of neurons per block in hidden layers between nn.BlockSparses.'},
      {arg='window_size', type='table', req=true,
       help='number of blocks used per example in each layer.'},
      {arg='gater_size', type='table', req=true,
       help='number of neurons in gater hidden layers'},
      {arg='output_size', type='number', req=true,
       help='output size of last layer'},
      {arg='noise_std', type='table', req=true,
       help='std deviation of gaussian noise used for NoisyReLU'},
      {arg='threshold_lr', type='number', default=0.1,
       help='learning rate to get the optimum threshold for a desired sparsity'},
      {arg='alpha_range', type='table', default='',
       help='{start_alpha, num_batches, endsall}'},
      {arg='sparse_init', type='boolean', default=false,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='typename', type='string', default='BlockSparse', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._gater_size = gater_size
   self._n_block = n_block
   self._hidden_size = hidden_size
   self._window_size = window_size
   alpha_range = (alpha_range == '') and {0.5, 1000, 0.01} or alpha_range
   
   require 'nnx'
      
   -- experts
   self._experts = {
      nn.BlockSparse(1, self._input_size, self._n_block[1], self._hidden_size[1]),
      nn.BlockSparse(self._n_block[1], self._hidden_size[1], self._n_block[2], self._hidden_size[2]),
      nn.BlockSparse(self._n_block[2], self._hidden_size[2], 1, self._output_size)
   }
   
   -- gaters
   self._gates = {
      nn.NoisyReLU(self._window_size[1]/self._n_block[1], alpha_range, threshold_lr, noise_stdv[1]),
      nn.NoisyReLU(self._window_size[2]/self._n_block[2], alpha_range, threshold_lr, noise_stdv[2])
   }
   self._gater = nn.Sequential()
   local inputSize = self._input_size
   for i, gater_size in ipairs(self._gater_size) do
      self._gater:add(nn.Linear(inputSize, self._gater_size[i]))
      self._gater:add(nn.Tanh())
      inputSize = self._gater_size[i]
   end
   local concat = nn.ConcatTable()
   local subGater1 = nn.Sequential()
   subGater1:add(nn.Linear(self._gater_size, self._n_block[1]))
   subGater1:add(self._gates[1])
   subGater1:add(nn.Sort(2,true))
   local para = nn.ParallelTable()
   para:add(nn.Narrow(2, 1, self._window_size[1]))
   para:add(nn.Narrow(2, 1, self._window_size[2]))
   subGater1:add(para)
   concat:add(subGater1)
   local subGater2 = nn.Sequential()
   subGater2:add(nn.Linear(self._gater_size, self._n_block[2]))
   subGater2:add(self._gates[2])
   subGater2:add(nn.Sort(2,true))
   subGater2:add(para:clone())
   concat:add(subGater2)
   self._gater:add(concat)
   
   -- mixture
   self._module = nn.BlockMixture(self._experts, self._gater)
   
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'bf'
   config.tags = config.tags or {}
   config.tags['no-momentum'] = true
   config.sparse_init = sparse_init
   parent.__init(self, config)
   -- only works with cuda
   self:type('torch.CudaTensor')
end

function BlockSparse:_forward(carry)
   self._gates[1].train = true
   self._gates[2].train = true
   self:outputAct(self._module:forward(self:inputAct()))
   local sparsityA = self._gates[1].sparsity
   local sparsityB = self._gates[2].sparsity
   self._stats.stdA = 0.9*self._stats.stdA + 0.1*sparsityA:std()
   self._stats.stdB = 0.9*self._stats.stdB + 0.1*sparsityB:std()
   self._stats.meanA = 0.9*self._stats.meanA + 0.1*sparsityA:mean()
   self._stats.meanB = 0.9*self._stats.meanB + 0.1*sparsityB:mean()
   return carry
end

function BlockSparse:_evaluate(carry)
   self._gates[1].train = false
   self._gates[2].train = false
   self:outputAct(self._module:forward(self:inputAct()))
   return carry
end

function BlockSparse:_backward(carry)
   self._acc_scale = carry.scale or 1
   self._report.scale = self._acc_scale
   local input_act = self:inputAct()
   local output_grad = self:outputGrad()
   -- we don't accGradParameters as updateParameters will do so inplace
   output_grad = self._module:updateGradInput(input_act, output_grad)
   self:inputGrad(output_grad)
   return carry
end

function BlockSparse:paramModule()
   return self._module
end

function BlockSparse:_type(type)
   self._input_type = type
   self._output_type = type
   self._module:type(type)
   return self
end

function BlockSparse:zeroGradParameters()
   --self._module:zeroGradParameters()
end

function BlockSparse:parameters()
   return self._module:parameters()
end

function BlockSparse:sharedClone()
   error"NotImplemented"
end

function BlockSparse:updateParameters(lr)
   -- we update parameters inplace (much faster)
   -- so don't use this with momentum (disabled by default)
   self._module:accUpdateGradParameters(self:inputAct(), self.outputGrad(), self._acc_scale * lr)
   return
end


-- Only affects 2D parameters.
-- Assumes that 2D parameters are arranged (output_dim x input_dim)
function BlockSparse:maxNorm(max_out_norm, max_in_norm)
   self._norm_iter = self._norm_iter + 1
   if self._norm_iter == self._norm_period then
      assert(self.backwarded, "Should call maxNorm after a backward pass")
      max_out_norm = self.mvstate.max_out_norm or max_out_norm
      max_in_norm = self.mvstate.max_in_norm or max_in_norm
      local params, gradParams = self:parameters()
      for k,param in pairs(params) do
         if param:dim() == 2 then
            if max_out_norm then
               -- rows feed into output neurons 
               param:renorm(1, 2, max_out_norm)
            end
            if max_in_norm then
               -- cols feed out from input neurons
               param:renorm(2, 2, max_out_norm)
            end
         end
      end
      self._norm_iter = 0
   end
end

function BlockSparse:_zeroStatistics()
   self._stats.stdA = 0
   self._stats.stdB = 0
   self._stats.meanA = 0
   self._stats.meanB = 0
end

function BlockSparse:report()
   print(self:name())
   print("mean+-std", self._stats.meanA, "+-", self._stats.stdA, self._stats.meanB, "+-", self._stats.stdB)
end
