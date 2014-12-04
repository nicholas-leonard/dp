------------------------------------------------------------------------
--[[ MaxNorm ]]--
-- Visitor
-- Hard constraint on the upper bound of the norm of output and/or input
-- neuron weights (in a weight matrix). Has a regularization effect 
-- analogous to WeightDecay, but with easier to optimize 
-- hyper-parameters. Quite useful with Rectified Linear Units (ReLU).
-- Should occur after Learn in VisitorChain
------------------------------------------------------------------------
local MaxNorm, parent = torch.class("dp.MaxNorm", "dp.Visitor")
MaxNorm.isMaxNorm = true

function MaxNorm:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, max_out_norm, max_in_norm, period, name = xlua.unpack(
      {config},
      'MaxNorm', 
      'Hard constraint on the upper bound of the norm of output ' ..
      'and input weights.',
      {arg='max_out_norm', type='number', default=1,
       help='max norm of output neuron weights'},
      {arg='max_in_norm', type='number', 
       help='max norm of input neuron weights'},
      {arg='period', type='number', default=1,
       help='Every period batches, the norm is constrained'},
      {arg='name', type='string', default='maxnorm',
       help='identifies visitor in reports.'}
   )
   self._max_out_norm = max_out_norm
   self._max_in_norm = max_in_norm
   self._period = period
   self._iter = 0
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-maxnorm')
   config.name = name
   parent.__init(self, config)
end

function MaxNorm:_visitModel(model)
   self._iter = self._iter + 1
   if self._iter == self._period then
      if model.maxNorm then
         model:maxNorm(self._max_out_norm, self._max_in_norm)
         return
      else
         if not model.mvstate[self:id():name()].warned then
            print("Warning: MaxNorm not implemented for model " .. 
               torch.typename(model) .. ". Ignoring model-visitor pair")
            model.mvstate[self:id():name()].warned = true
         end
      end
      self._iter = 0
   end
end
