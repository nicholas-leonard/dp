------------------------------------------------------------------------
--[[ Dictionary ]]-- 
-- Adapts a nn.LookupTable
-- Works on a WordTensor:context() view.
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
   self._module = nn.LookupTable(dict_size, output_size)
   config.typename = typename
   config.input_type = 'torch.IntTensor'
   config.tags = config.tags or {}
   config.input_view = 'bt'
   config.output_view = 'bwc'
   config.output = dp.SequenceView()
   parent.__init(self, config)
end

function Dictionary:_forward(carry)
   local activation = self:inputAct()
   activation = self._module:forward(activation)
   self:outputAct(activation)
   return carry
end

function Dictionary:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local output_grad = self:outputGrad()
   local input_act = self:inputAct()
   self._module:backward(input_act, output_grad, scale)
   return carry
end

function Dictionary:zeroGradParameters()
   self._module:zeroGradParameters()
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

function Dictionary:reset()
   self._module:reset()
end

function Dictionary:parameters()
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

function Dictionary:share(dict, ...)
   assert(dict.isDictionary)
   return parent.share(self, dict, ...)
end

-- Only affects 2D parameters.
-- Assumes that 2D parameters are arranged (input_dim, output_dim)
function Dictionary:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   local module = self._module
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_out_norm = self.mvstate.max_in_norm or max_in_norm or max_out_norm
   for k,nBackward in pairs(module.inputs) do
      module.weight:narrow(1, k, 1):renorm(1, 2, max_out_norm)
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
   clone._module.gradWeight:resizeAs(self._module.gradWeight)
   clone._module.batchSize = self._module.batchSize
   return self:share(clone, 'weight')
end

function Dictionary:paramModule()
   return self._module
end
