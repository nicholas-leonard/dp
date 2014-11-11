------------------------------------------------------------------------
--[[ RecurrentVisitorChain ]]--
-- Composite, Visitor, Chain of Responsibility
-- Used by Recurrent Neural Networks to visit sequences of batches
------------------------------------------------------------------------
local RecurrentVisitorChain, parent = torch.class("dp.RecurrentVisitorChain", "dp.VisitorChain")
RecurrentVisitorChain.isRecurrentVisitorChain = true

function RecurrentVisitorChain:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, visit_interval = xlua.unpack(
      {config},
      'RecurrentVisitorChain', 
      'A composite chain of visitors used to visit recurrent models. '..
      'Subscribes to Mediator channel "doneSequence", which when '..
      'published to, forces a visit. Otherwise, visits models every '..
      'visit_interval epochs.',
      {arg='visit_interval', type='number', req=true,
       help='model is visited every visit_interval epochs'}
   )
   config.name = config.name or 'recurrentvisitorchain'
   parent.__init(self, config)
   self._visit_interval = visit_interval
   self._n_visit = 0
end

function RecurrentVisitorChain:setup(config)
   parent.setup(self, config)
   -- published by SentenceSampler
   self._mediator:subscribe('doneSequence', self, 'doneSequence')
end

function RecurrentVisitorChain:doneSequence()
   self._force_visit = true
end

function RecurrentVisitorChain:_visitModel(model)
   self._n_visit = self._n_visit + 1
   if self._force_visit or self._n_visit == self._visit_interval then
      parent._visitModel(self, model)
      self._force_visit = false
   end
end
