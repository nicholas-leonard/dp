------------------------------------------------------------------------
--[[ RecurrentDictionary ]]-- 
-- A Simple Recurrent Neural Network used for Language Modeling
-- Adapts a nn.Recurrent containing LookupTable (input) and
-- Linear (feedback) Modules. Should be located at the input of the 
-- computational flow graph (has no gradInputs).
-- Updates are accumulated in-place (acc_update = true).
------------------------------------------------------------------------
local RecurrentDictionary, parent = torch.class("dp.RecurrentDictionary", "dp.Layer")
RecurrentDictionary.isRecurrentDictionary = true

function RecurrentDictionary:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, dict_size, output_size, transfer, typename
      = xlua.unpack(
      {config},
      'RecurrentDictionary', 
      'adapts a nn.Recurrent containing LookupTable (input) and '..
      'Linear (feedback) Modules.',
      {arg='dict_size', type='number', req=true,
       help='Number of entries in the dictionary (e.g. num of words)'},
      {arg='output_size', type='number', req=true,
       help='Number of neurons per entry. Also the size of the '..
       'input and output size of the feedback layer'},
      {arg='transfer', type='nn.Module',
       help='a transfer function like nn.Tanh, nn.Sigmoid, etc.'..
       'Defaults to nn.Sigmoid (recommended for RNNs)'},
      {arg='typename', type='string', default='recurrentdictionary', 
       help='identifies Model type in reports.'}
       
   )
   assert(not config.dropout, 
      "RecurrentDictionary doesn't work with dropout (maybe later)")
   assert(not config.sparse_init, 
      "RecurrentDictionary doesn't work with sparse_init (maybe later)")
   config.acc_update = true -- so momentum and cie. aren't performed
   config.sparse_init = false
   self._dict_size = dict_size
   self._output_size = output_size
   self._transfer = transfer or nn.Sigmoid()
   self._lookup = nn.LookupTable(dict_size, output_size)
   -- by default, we backwardUpdateThroughTime, so delete gradWeights :
   self._lookup:accUpdateOnly()
   self._feedback = nn.Linear(output_size, output_size)
   self._recurrent = nn.Recurrent(
      output_size, self._lookup, self._feedback, self._transfer
   )
   self._module = self._recurrent
   config.typename = typename
   config.input_type = 'torch.IntTensor'
   config.tags = config.tags or {}
   config.input_view = 'b' -- input is a batch of indices
   config.output_view = 'bf'
   config.output = dp.DataView()
   parent.__init(self, config)
end

function RecurrentDictionary:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe('beginSequence', self, 'beginSequence')
end

function RecurrentDictionary:beginSequence()
   self._recurrent:forget() -- forget the current sequence, start anew
end

function RecurrentDictionary:_backward(carry)
   self._module:backward(self:inputAct(), self:outputGrad(), self._acc_scale)
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
   local params, gradParams = self._module:parameters()
   local scales
   if self.forwarded then
      local lookupParams = self._lookup:parameters()
      for i, param in ipairs(lookupParams) do
         local idx = _.indexOf(params, param)
         table.remove(params, idx)
         table.remove(gradParams, idx)
      end
      local offset = #params
      scales = {}
      -- only return the parameters affected by the forward/backward
      for k,nBackward in pairs(self._lookup.inputs) do
         local kscale = self._lookup:scaleUpdateByKey(k)
         params[offset+k] = self._lookup.weight:select(1, k)
         gradParams[offset+k] = self._lookup.gradWeight:select(1, k)
         scales[offset+k] = self._lookup:scaleUpdateByKey(k)
      end
   end
   return params, gradParams, scales
end

function RecurrentDictionary:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_out_norm = self.mvstate.max_in_norm or max_in_norm or max_out_norm
   
   for k,nBackward in pairs(self._lookup.inputs) do
      self._lookup.weight:narrow(1, k, 1):renorm(1, 2, max_out_norm)
   end
   
   local lookupParams = self._lookup:parameters()
   local params, gradParams = self._module:parameters()
   for k,param in pairs(params) do
      if param:dim() == 2 and not _.contains(lookupParams, param) then
         if max_out_norm then
            -- rows feed into output neurons 
            param:renorm(1, 2, max_out_norm)
         end
         if max_in_norm then
            -- cols feed out from input neurons
            param:renorm(2, 2, max_in_norm)
         end
      end
   end
end

function RecurrentDictionary:updateParameters(lr)
   self._module:updateParameters(lr)
end

function RecurrentDictionary:share(rnn, ...)
   assert(rnn.isRecurrentDictionary)
   return parent.share(self, rnn, ...)
end
