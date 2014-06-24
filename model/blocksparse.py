------------------------------------------------------------------------
--[[ WindowSparse ]]--
-- A 3 layer model of Distributed Conditional Computation
------------------------------------------------------------------------
local BlockSparse, parent = torch.class("dp.BlockSparse", "dp.Layer")
BlockSparse.isBlockSparse = true

function BlockSparse:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, hidden_size, gater_size, output_size, 
      lr, sparse_init, typename, sparsityFactor, threshold_lr, alpha_range, std
      = xlua.unpack(
      {config},
      'BlockSparse', 
      'Deep mixture of experts model. It is three parametrized '..
      'layers of gated experts. There are two gaters, one for each '..
      'layer of hidden neurons.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hidden_size', type='table', req=true,
       help='size of the hidden layers between nn.BlockSparses.'},
      {arg='gater_size', type='table', req=true,
       help='output size of the gaters'},
      {arg='output_size', type='number', req=true,
       help='output size of last layer'},
      {arg='lr', type='table', req=true,
       help='lr of gaussian blur target'},
      {arg='sparse_init', type='boolean', default=false,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'}
      {arg='typename', type='string', default='BlockSparse', 
       help='identifies Model type in reports.'},
      {arg='sparsityFactor', type='number', default=0.1,
       help='sparsity for noisyrelu'},
      {arg='threshold_lr', type='number', default=0.1,
       help='learning rate to get the optimum threshold for a desired sparsity'},
      {arg='alpha_range', type='table', default={0.5, 1000, 0.01},
       help='{start_alpha, num_batches, endsall}'},
      {arg='std', type='number', req=true, default=0.1,
       help='std for gaussian noise'},
   )
   self._input_size = input_size
   self._output_size = output_size
   self._gater_size = gater_size
   self._hidden_size = hidden_size
   self._lr = lr
   
   
   require 'cunnx'
      
   --[[ First Layer : Input is dense, output is sparse ]]--
   -- Gater A
   local gaterA = nn.Sequential()
   gaterA:add(nn.Linear(self._input_size, self._gater_size[1]))
   local gateA = nn.NoisyReLU(opt.sparsityFactor, opt.threshold_lr, opt.alpha_range, opt.std)
   gaterA:add(gateA)
   gaterA:add(nn.SortFilter(opt.sparsityFactor))
   
   -- Mixture of experts A
   local concatA = nn.ConcatTable() -- outputs a table of tensors
   concatA:add(nn.Identity()) -- forwards input as is
   concatA:add(gaterA) -- forwards gated expert indices
   
   local mixtureA = nn.Sequential()
   mixtureA:add(concatA)
   expertsA = nn.BlockSparse(self._input_size, self._hidden_size[1], 1, self._gater_size[1])
   mixtureA:add(expertsA)
   mixtureA:add(nn.Tanh)
   

   --[[ Second Layer : Input and output are sparse ]]--
   -- Gater B : The input to the B gater is sparse so we use a nn.BlockSparse instead of a nn.Linear:
   local gaterB = nn.Sequential()
   gaterB:add(nn.BlockSparse(self._hidden_size[1]), self._gater_size[2], self._gater_size[1], 1))
   gateB = nn.NoisyReLU(opt.sparsityFactor, opt.threshold_lr, opt.alpha_range, opt.std)
   gaterB:add(gateB)
   gaterB:add(nn.SortFilter(opt.sparsityFactor))

   
   -- Mixture of experts B
   local concatB = nn.ConcatTable() -- outputs a table of tensors
   concatB:add(nn.Identity()) -- forwards the input to the output
   concatB:add(gaterB)

   local mixtureB = nn.Sequential()
   mixtureB:add(concatB)
   expertsB = nn.BlockSparse(self._hidden_size[1]), self._gater_size[2], self._gater_size[1], self._gater_size[2])
   mixtureB:add(expertsB)
   mixture:add(nn.Tanh)

   --[[ Third Layer : Input is sparse, output is dense ]]--
   -- Mixture of experts C
   local mixtureC = nn.Sequential()
   expertsC = nn.BlockSparse(self._hidden_size[2], self._output_size, self._gater_size[2], 1)
   mixtures:add(expertsC)
   mixtureC:add(nn.Tanh())

   --[[ Stack Mixtures ]]--
   -- Stack the 3 mixtures of experts layers. 
   local mlp = nn.Sequential()
   mlp:add(mixtureA)
   mlp:add(mixtureB)
   mlp:add(mixtureC)
   
   self._gaterA = gaterA
   self._gaterB = gaterB
   self._gateA = gateA
   self._gateB = gateB
   self._expertsA = expertsA
   self._expertsB = expertsB
   self._expertsC = expertsC
   self._mixtureA = mixtureA
   self._mixtureB = mixtureB
   self._mixtureC = mixtureC
   self._module = mlp
   
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'bf'
   config.tags = config.tags or {}
   config.tags['no-momentum'] = true
   config.sparse_init = sparse_init
   parent.__init(self, config)
end

-- requires targets be in carry
function BlockSparse:_forward(carry)
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

function BlockSparse:reset()
   self._module:reset()
   if self._sparse_init then
      self._sparseReset(self._module.weight)
   end
end

function BlockSparse:zeroGradParameters()
   self._module:zeroGradParameters(true)
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function BlockSparse:parameters()
   return self._module:parameters(true)
end

function BlockSparse:sharedClone()
   local clone = torch.protoClone(self, {
      input_size=self._input_size, hierarchy={[1]=torch.IntTensor{1,2}},
      root_id=1, sparse_init=self._sparse_init,
      dropout=self._dropout and self._dropout:clone(),
      typename=self._typename, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   })
   clone._target_type = self._target_type
   clone._module = self._module:sharedClone()
   return clone
end

function BlockSparse:updateParameters(lr)
   -- we update parameters inplace (much faster)
   -- so don't use this with momentum (disabled by default)
   self._module:accUpdateGradParameters(self:inputAct(), self.outputGrad(), self._acc_scale * lr)
   return
end

