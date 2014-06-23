------------------------------------------------------------------------
--[[ WindowSparse ]]--
-- A 3 layer model of Distributed Conditiona Computation
------------------------------------------------------------------------
local WindowSparse, parent = torch.class("dp.WindowSparse", "dp.Layer")
WindowSparse.isWindowSparse = true

function WindowSparse:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, window_size, input_stdv, output_stdv, 
      hidden_size, gater_size, output_size, typename, lr, maxOutNorm 
      = xlua.unpack(
      {config},
      'WindowSparse', 
      'Deep mixture of experts model. It is three parametrized '..
      'layers of gated experts. There are two gaters, one for each '..
      'layer of hidden neurons.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='window_size', type='table', req=true,
       help='A table of window sizes. Should be smaller than '..
       'commensurate hidden size.'},
      {arg='inputStdv', type='table', req=true,
       help='stdv of gaussian blur of targets for WindowGate'},
      {arg='outputStdv', type='table', req=true,
       help='stdv of gaussian blur of output of WindowGate'},
      {arg='hidden_size', type='table', req=true,
       help='size of the hidden layers between nn.WindowSparses.'},
      {arg='gater_size', type='table', req=true,
       help='output size of the gaters'},
      {arg='output_size', type='number', req=true,
       help='output size of last layer'},
      {arg='lr', type='table', req=true,
       help='lr of gaussian blur target'},
      {arg='typename', type='string', default='windowsparse', 
       help='identifies Model type in reports.'},
      {arg='maxOutNorm', type='number', default=1,
       help='max norm of output neuron weights. '..
       'Overrides MaxNorm visitor'}
   )
   self._input_size = input_size
   self._window_size = window_size
   self._output_size = output_size
   self._gater_size = gater_size
   self._hidden_size = hidden_size
   self._input_stdv = input_stdv
   self._output_stdv = output_stdv
   self._lr = lr
   
   require 'cunnx'
      
   --[[ First Layer : Input is dense, output is sparse ]]--
   -- Gater A
   local gaterA = nn.Sequential()
   gaterA:add(nn.Linear(self._input_size, self._gater_size[1]))
   gaterA:add(nn.Softmax())
   local gateA = nn.WindowGate(self._window_size[1], self._hidden_size[1], self._input_stdv[1], self._output_stdv[1], self._lr[1])
   gaterA:add(gateA)
   
   -- Mixture of experts A
   local concatA = nn.ConcatTable() -- outputs a table of tensors
   concatA:add(nn.Identity()) -- forwards input as is
   concatA:add(gaterA) -- forwards gated expert indices
   local mixtureA = nn.Sequential()
   mixtureA:add(concatA)
   
   -- experts :
   local expertsA = nn.WindowSparse(self._input_size, self._hidden_size[1], nn.WindowSparse.DENSE_SPARSE)
   mixtureA:add(expertsA) 
   local paraA = nn.ParallelTable()
   paraA:add(nn.Tanh()) -- non-linearity of experts (WindowSparse)
   paraA:add(nn.Identity())
   mixtureA:add(paraA)

   --[[ Second Layer : Input and output are sparse ]]--
   -- Gater B : The input to the B gater is sparse so we use a nn.WindowSparse instead of a nn.Linear:
   local gaterB = nn.Sequential()
   gaterB:add(nn.WindowSparse(self._hidden_size[1] ,self._gater_size[2], nn.WindowSparse.SPARSE_DENSE)) 
   gaterB:add(nn.SoftMax())
   local gateB = nn.WindowGate(self._window_size[2], self._hidden_size[2], self._input_stdv[2], self._output_stdv[2], self._lr[2])
   gaterB:add(gateB)
   
   -- Mixture of experts B
   local concatB = nn.ConcatTable() -- outputs a table of tensors
   concatB:add(nn.Identity()) -- forwards the input to the output
   concatB:add(gaterB)
   local mixtureB = nn.Sequential()
   mixtureB:add(concatB)
   
   -- experts
   -- this is where most of the sparsity stems from :
   local expertsB = nn.WindowSparse(self._hidden_size[1], self._hidden_size[2], nn.WindowSparse.SPARSE_SPARSE)
   mixtureB:add(expertsB)
   local paraB = nn.ParallelTable()
   paraB:add(nn.Tanh()) -- non-linearity of experts (WindowSparse)
   paraB:add(nn.Identity())
   mixtureB:add(paraA)

   --[[ Third Layer : Input is sparse, output is dense ]]--
   -- Mixture of experts C
   local mixtureC = nn.Sequential()
   local expertsC = nn.WindowSparse(self._hidden_size[2], self._output_size, nn.WindowSparse.SPARSE_DENSE)
   mixtureC:add(expertsC)
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
   config.output_view = 'b'
   config.tags = config.tags or {}
   config.tags['no-maxnorm'] = true
   parent.__init(self, config)
   self._target_type = 'torch.IntTensor'
end

-- requires targets be in carry
function WindowSparse:_forward(carry)
   local activation = self:inputAct()
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not carry.evaluate)
      activation = self._dropout:forward(activation)
      self.mvstate.dropoutAct = activation
   end
   assert(carry.targets and carry.targets.isClassView,
      "carry.targets should refer to a ClassView of targets")
   local targets = carry.targets:forward('b', self._target_type)
   -- outputs a column vector of likelihoods of targets
   activation = self._module:forward{activation, targets}
   self:outputAct(activation)
   return carry
end

function WindowSparse:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.dropoutAct or self:inputAct()
   local output_grad = self:outputGrad()
   assert(carry.targets and carry.targets.isClassView,
      "carry.targets should refer to a ClassView of targets")
   local targets = carry.targets:forward('b', self._target_type)
   output_grad = self._module:backward({input_act, targets}, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self:inputAct()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self:inputGrad(output_grad)
   return carry
end

function WindowSparse:paramModule()
   return self._module
end

function WindowSparse:_type(type)
   self._input_type = type
   self._output_type = type
   if self._dropout then
      self._dropout:type(type)
   end
   if type == 'torch.CudaTensor' then
      require 'cunnx'
      self._target_type = 'torch.CudaTensor'
   else
      self._target_type = 'torch.IntTensor'
   end
   self._module:type(type)
   return self
end

function WindowSparse:reset()
   self._module:reset()
   if self._sparse_init then
      self._sparseReset(self._module.weight)
   end
end

function WindowSparse:zeroGradParameters()
   self._module:zeroGradParameters(true)
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function WindowSparse:parameters()
   return self._module:parameters(true)
end

function WindowSparse:sharedClone()
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

function WindowSparse:updateParameters(lr)
   self._module:updateParameters(lr, true)
end

function WindowSparse:maxNorm()
   error"NotImplemented"
end

