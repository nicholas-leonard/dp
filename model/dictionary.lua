------------------------------------------------------------------------
--[[ Dictionary ]]-- 
-- Adapts a nn.LookupTable (often used for language modeling)
-- Outputs a SequenceView
------------------------------------------------------------------------
local Dictionary, parent = torch.class("dp.Dictionary", "dp.Layer")
Dictionary.isDictionary = true

function Dictionary:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, dict_size, output_size, typename
      = xlua.unpack(
      {config},
      'Dictionary', 
      'adapts a nn.LookupTable',
      {arg='dict_size', type='number', req=true,
       help='Number of entries in the dictionary (e.g. num of words)'},
      {arg='output_size', type='number', req=true,
       help='Number of neurons per entry.'},
      {arg='typename', type='string', default='dictionary', 
       help='identifies Model type in reports.'}
   )
   assert(not config.dropout, 
      "Dictionary doesn't work with dropout")
   assert(not config.sparse_init, 
      "Dictionary doesn't work with sparse_init")
   config.sparse_init = false
   self._dict_size = dict_size
   self._output_size = output_size
   self._lookup = nn.LookupTable(dict_size, output_size)
   self._module = self._lookup
   config.typename = typename
   config.input_type = 'torch.IntTensor'
   config.tags = config.tags or {}
   config.input_view = 'bt'
   config.output_view = 'bwc'
   config.output = dp.SequenceView()
   parent.__init(self, config)
   if self._acc_update then
      self._lookup:accUpdateOnly()
   end
end

function Dictionary:_backward(carry)
   local input_grad
   if self._acc_update then 
      input_grad = self._module:updateGradInput(self:inputAct(), self:outputGrad())
   else
      input_grad = self._module:backward(self:inputAct(), self:outputGrad(), self._acc_scale)
   end
   return carry
end

function Dictionary:_type(type)
   self._module:type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' or type == 'torch.CudaTensor' then
      self._output_type = type
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      self._input_type = type
   end
   return self
end

function Dictionary:reset(stdv)
   self._module:reset(stdv)
end

function Dictionary:parameters()
   if self.forwarded then
      -- only return the parameters affected by the forward/backward
      local params, gradParams, scales = {}, {}, {}
      for k,nBackward in pairs(self._lookup.inputs) do
         local kscale = self._lookup:scaleUpdateByKey(k)
         params[k] = self._lookup.weight:select(1, k)
         if not self._acc_update then
            gradParams[k] = self._lookup.gradWeight:select(1, k)
         end
         scales[k] = self._lookup:scaleUpdateByKey(k)
      end
      return params, gradParams, scales
   end
   return self._lookup:parameters()
end

function Dictionary:zeroGradParameters()
   self._lookup:zeroGradParameters() -- to reset the inputs table
end

function Dictionary:share(dict, ...)
   assert(dict.isDictionary)
   return parent.share(self, dict, ...)
end

function Dictionary:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_out_norm = self.mvstate.max_in_norm or max_in_norm or max_out_norm
   for k,nBackward in pairs(self._lookup.inputs) do
      self._lookup.weight:narrow(1, k, 1):renorm(1, 2, max_out_norm)
   end
end

function Dictionary:sharedClone()
   local clone = torch.protoClone(self, {
      dict_size = 1, output_size = 1, typename=self._typename, 
      gather_stats=self._gather_stats, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   })
   clone._dict_size = self._dict_size
   clone._output_size = self._output_size
   if self._acc_update then
      clone._lookup.gradWeight:resizeAs(self._lookup.gradWeight)
   end
   clone._lookup.batchSize = self._lookup.batchSize
   return self:share(clone, 'weight')
end
