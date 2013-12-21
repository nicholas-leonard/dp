------------------------------------------------------------------------
--[[ MaxNorm ]]--
-- ModelVisitor
-- Hard constraint on the upper bound of the norm of output and/or input
-- neuron weights (in a weight matrix). Has a regularization effect 
-- analogous to WeightDecay, but with easier to optimize 
-- hyper-parameters. Quite useful with Rectified Linear Units (ReLU).
-- Should occur after LearningRate in VisitorChain
------------------------------------------------------------------------

local MaxNorm, parent = torch.class("dp.MaxNorm", "dp.Visitor")

function MaxNorm:__init(config)
   config = config or {}
   local args, max_out_norm, max_in_norm, name = xlua.unpack(
      {config},
      'MaxNorm', 
      'Hard constraint on the upper bound of the norm of output ' ..
      'and input weights.',
      {arg='max_out_norm', type='number', default=1,
       help='max norm of output neuron weights'},
      {arg='max_in_norm', type='number', 
      help='max norm of input neuron weights'},
      {arg='name', type='string', default='maxnorm',
       help='identifies visitor in reports.'}
   )
   self._max_out_norm = max_out_norm
   self._max_in_norm = max_in_norm
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.name = name
   parent.__init(self, config)
end

function MaxNorm:_visitModel(model)
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
end
