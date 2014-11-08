------------------------------------------------------------------------
--[[ RecurrentDictionary ]]-- 
-- Adapts a nn.LookupTable
-- Works on a WordTensor:context() view.
------------------------------------------------------------------------
local RecurrentDictionary, parent = torch.class("dp.RecurrentDictionary", "dp.Layer")
RecurrentDictionary.isRecurrentDictionary = true

function RecurrentDictionary:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, dict_size, output_size, transfer, typename
      = xlua.unpack(
      {config},
      'RecurrentDictionary', 
      'adapts a nn.LookupTable',
      {arg='dict_size', type='number', req=true,
       help='Number of entries in the dictionary (e.g. num of words)'},
      {arg='output_size', type='number', req=true,
       help='Number of neurons per entry. Also the size of the '..
       'input and output size of the feedback layer'},
      {arg='transfer', type='nn.Module',
       help='a transfer function like nn.Tanh, nn.Sigmoid, etc.'..
       'Defaults to nn.Sigmoid (recommended for RNNs)'},
      {arg='typename', type='string', default='dictionary', 
       help='identifies Model type in reports.'}
       
   )
   assert(not config.dropout, 
      "RecurrentDictionary doesn't work with dropout (maybe later)")
   assert(not config.acc_update, 
      "RecurrentDictionary doesn't work with acc_update (maybe later)")
   assert(not config.sparse_init, 
      "RecurrentDictionary doesn't work with sparse_init (maybe later)")
   config.acc_update = false
   config.sparse_init = false
   self._dict_size = dict_size
   self._output_size = output_size
   self._transfer = transfer or nn.Sigmoid()
   self._lookup = nn.LookupTable(dict_size, output_size)
   self._feedback = nn.Linear(outputSize, outputSize)
   self._recurrent = nn.Recurrent(
      dict_size, self._lookup, self._feedback, self._transfer
   )
   self._module = self._recurrent
   config.typename = typename
   config.input_type = 'torch.IntTensor'
   config.tags = config.tags or {}
   config.input_view = 'bt'
   config.output_view = 'bf'
   config.output = dp.SequenceView()
   parent.__init(self, config)
end

function RecurrentDictionary:_backward(carry)
   local input_grad = self._module:backward(self:inputAct(), self:outputGrad(), self._acc_scale)
   return carry
end

function RecurrentDictionary:_type(type)
   self._module:type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor' then
      self._output_type = type
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      self._input_type = type
   end
   return self
end

function RecurrentDictionary:reset(stdv)
   self._module:reset(stdv)
end

function RecurrentDictionary:parameters()
   error"Not Implemented"
   local module = self._module
   if self.forwarded then
      -- only return the parameters affected by the forward/backward
      local params, gradParams, scales = {}, {}, {}
      for k,nBackward in pairs(module.inputs) do
         local kscale = module:scaleUpdateByKey(k)
         params[k] = module.weight:select(1, k)
         gradParams[k] = module.gradWeight:select(1, k)
         scales[k] = module:scaleUpdateByKey(k)
      end
      return params, gradParams, scales
   end
   return module:parameters()
end

function RecurrentDictionary:share(rnn, ...)
   assert(rnn.isRecurrentDictionary)
   return parent.share(self, rnn, ...)
end

-- Only affects 2D parameters.
-- Assumes that 2D parameters are arranged (input_dim, output_dim)
function RecurrentDictionary:maxNorm(max_out_norm, max_in_norm)
   error"Not Implemented"
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   local module = self._module
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_out_norm = self.mvstate.max_in_norm or max_in_norm or max_out_norm
   for k,nBackward in pairs(module.inputs) do
      module.weight:narrow(1, k, 1):renorm(1, 2, max_out_norm)
   end
end
