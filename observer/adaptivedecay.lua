------------------------------------------------------------------------
--[[ AdaptiveDecay ]]--
-- Observer used by optimizer callbacks.
-- Decays decay attribute when error on validation doesn't reach a new 
-- minima for max_wait epochs.
-- Should observe in conjuction with a dp.ErrorMinima instance (such as 
-- EarlyStopper)
------------------------------------------------------------------------
local AdaptiveDecay, parent = torch.class("dp.AdaptiveDecay", "dp.Observer")

function AdaptiveDecay:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args
   args, self._max_wait, self._decay_factor 
      = xlua.unpack(
      {config},
      'AdaptiveDecay', 
      'Decays learning rate when validation set does not reach '..
      'a new minima for max_wait epochs',
      {arg='max_wait', type='number', default=2,
       help='maximum number of epochs to wait for a new minima ' ..
       'to be found. After that, the learning rate is decayed.'},
      {arg='decay_factor', type='number', default=0.1,
       help='Learning rate is decayed by lr = lr*decay_factor every '..
       'time a new minima has not been reached for max_wait epochs'}
   )
   parent.__init(self, "errorMinima")
   self._wait = 0
   -- public attribute
   self.decay = 1
end

function AdaptiveDecay:errorMinima(found_minima)
   self._wait = found_minima and 0 or self._wait + 1
   if self._max_wait < self._wait then
      self._wait = 0  
      self.decay = self._decay_factor
   else
      self.decay = 1
   end
end


