------------------------------------------------------------------------
--[[ Loss ]]--
-- Node subclass
-- Adapter of nn.Criterion
------------------------------------------------------------------------
local Loss, parent = torch.class("dp.Loss", "dp.Node")
Loss.isLoss = true

function Loss:__init()

end

function Loss:setTargetState(tstate)
   assert(tstate, "No Target State")
   if tstate.isBaseTensor then
      -- tstate is BaseTensor, assume it represents targets
      tstate = {target=tstate}
   end
   assert(type(tstate) == 'table')
   self.tstate = tstate
end

function Loss:forward(state)
   self:setTargetState(state.target)
   local loss, cstate = parent.forward(self, state)
   self:updateLoss(loss)
   return loss, cstate
end

function Loss:evaluate(state)
   self:setTargetState(state.target)
   local loss, cstate = parent.evaluate(self, state)
   self:updateLoss(loss)
   return loss, cstate
end

function Loss:backward(state)
   self:setTargetState(state.target)
   self:setGlobalState(state.global)
   local cstate = self:_backward(table.copy(state.carry)) or state.carry
   self.backwarded = true
   return self.istate, cstate
end

function Loss:_forward(cstate)
end

function Loss:_backward(cstate)
end

function Loss:doneBatch(...)
   parent.doneBatch(self, ...)
   self.tstate = {} -- target state
end

function Loss:updateLoss(loss)
   self._loss = self._loss + batch:loss()                
   self._samples_seen = self._samples_seen + self.gstate.n_sample
end

function Loss:reset()
   self._loss = 0
   self._samples_seen = 0
end

function Loss:loss()
   return self._loss / self._samples_seen
end
