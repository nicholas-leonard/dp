------------------------------------------------------------------------
--[[ NarrowDictionary ]]-- 
-- Adapts a nn.NarrowLookupTable
------------------------------------------------------------------------
local NarrowDictionary, parent = torch.class("dp.NarrowDictionary", "dp.Dictionary")
NarrowDictionary.isNarrowDictionary = true

function NarrowDictionary:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, dict_size, output_size, delta_size, typename
      = xlua.unpack(
      {config},
      'NarrowDictionary', 
      'adapts a nn.LookupTable',
      {arg='dict_size', type='number', req=true,
       help='Number of entries in the dictionary (e.g. num of words)'},
      {arg='output_size', type='number', req=true,
       help='Number of neurons per entry.'},
      {arg='delta_size', type='number', req=true,
       help='Size that each successive item in the context looses'},
      {arg='typename', type='string', default='dictionary', 
       help='identifies Model type in reports.'}
   )
   assert(not config.dropout, 
      "NarrowDictionary doesn't work with dropout")
   assert(not config.sparse_init, 
      "NarrowDictionary doesn't work with sparse_init")
   config.sparse_init = false
   self._dict_size = dict_size
   self._output_size = output_size
   self._delta_size = delta_size
   self._module = nn.NarrowLookupTable(delta_size, dict_size, output_size, true)
   if self._acc_update then
      self._module:accUpdateOnly()
   end
   config.typename = typename
   config.input_type = 'torch.IntTensor'
   config.tags = config.tags or {}
   config.input_view = 'bt'
   config.output_view = 'bf'
   config.output = dp.DataView()
   dp.Layer.__init(self, config)
end

function NarrowDictionary:sharedClone()
   local clone = torch.protoClone(self, {
      dict_size = 1, output_size = 1, delta_size = 1, 
      typename=self._typename, gather_stats=self._gather_stats, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   })
   clone._dict_size = self._dict_size
   clone._output_size = self._output_size
   clone._delta_size = self._delta_size
   if self._acc_update then
      clone._module.gradWeight:resizeAs(self._module.gradWeight)
   end
   clone._module.batchSize = self._module.batchSize
   return self:share(clone, 'weight')
end
