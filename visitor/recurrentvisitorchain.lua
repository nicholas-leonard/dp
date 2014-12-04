------------------------------------------------------------------------
--[[ RecurrentVisitorChain ]]--
-- Composite, Visitor, Chain of Responsibility
-- Used by Recurrent Neural Networks to visit sequences of batches
------------------------------------------------------------------------
local RecurrentVisitorChain, parent = torch.class("dp.RecurrentVisitorChain", "dp.VisitorChain")
RecurrentVisitorChain.isRecurrentVisitorChain = true

function RecurrentVisitorChain:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, visit_interval, force_forget = xlua.unpack(
      {config},
      'RecurrentVisitorChain', 
      'A composite chain of visitors used to visit recurrent models. '..
      'Subscribes to Mediator channel "doneSequence", which, when '..
      'notified, forces a visit. Otherwise, visits models every '..
      'visit_interval epochs.',
      {arg='visit_interval', type='number', req=true,
       help='model is visited every visit_interval epochs'},
      {arg='force_forget', type='boolean', default=false,
       help='force recurrent models to forget after each update'} 
   )
   config.name = config.name or 'recurrentvisitorchain'
   parent.__init(self, config)
   self._visit_interval = visit_interval
   self._force_forget = force_forget
   self._force_visit = false
   self._n_visit = 0
end

function RecurrentVisitorChain:setup(config)
   parent.setup(self, config)
   -- published by SentenceSampler
   self._mediator:subscribe('doneSequence', self, 'doneSequence')
   self._mediator:subscribe('doneEpoch', self, 'doneEpoch')
end

function RecurrentVisitorChain:doneSequence()
   self._force_visit = true
end

function RecurrentVisitorChain:doneEpoch()
   self._n_visit = 0
end

function RecurrentVisitorChain:_visitModel(model)
   self._n_visit = self._n_visit + 1
   if self._force_visit or self._n_visit == self._visit_interval then
      parent._visitModel(self, model)
      if self._force_forget and model.forget then
         model:forget()
      end
      self._n_visit = 0
      self._force_visit = false
      self._done_visit = true
   end
end

function RecurrentVisitorChain:doneVisit(model)
   -- only zeroGradParameters when the model is actually visited
   if self._zero_grads and self._done_visit then
      model:zeroGradParameters()
      self._done_visit = false
   end
end
