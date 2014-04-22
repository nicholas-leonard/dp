------------------------------------------------------------------------
--[[ Command ]]--
-- Command design pattern
-- Serializable
-- An object that can be sent over the network
------------------------------------------------------------------------
local Command = torch.class("dp.Command")

function Command.__init(subject_id)
   self._subject_id = subject_id
end

function Command:subjectId()
   return self._subject_id
end

function Command:execute(session)
   error"Not Implemented"
end

------------------------------------------------------------------------
--[[ Forward ]]--
------------------------------------------------------------------------
local Forward, parent = torch.class("dp.Forward", "dp.Command")

function Forward:__init(model_id, input_state, carry_state, batch_state)
   parent.init(model_id)
   self._input_state = input_state
   self._carry_state = carry_state
   self._batch_state = batch_state
end

function Forward:execute(session)
   local session_map, shared_map = session:objectMaps()
   local output_state, carry_state = model:forward(
      self._input_state, self._carry_state, self._batch_state
   )
   return output_state, carry_state
end

------------------------------------------------------------------------
--[[ Locate ]]--
-- Used by stations to query master to locate remote objects
------------------------------------------------------------------------
local Locate, parent = torch.class("dp.Locate", "dp.Command")

function Locate:__init(master_id, addr)
   parent.init(master_id)
   self._addr = addr
end

function Locate:execute(session)
   local session_map, shared_map = session:objectMaps()
   local model = getSharedWithLocalMemento(self._subject_id)
   local output_state, carry_state = model:forward(
      self._input_state, self._carry_state, self._batch_state
   )
   return output_state, carry_state
end

------------------------------------------------------------------------
--[[ NewSession ]]--
-- Used by stations to query master to locate remote objects
------------------------------------------------------------------------
local NewSession, parent = torch.class("dp.NewSession", "dp.Command")



------------------------------------------------------------------------
--[[ PropagateBatch ]]--
------------------------------------------------------------------------
local PropagateBatch, parent = torch.class("dp.PropagateBatch", "dp.Command")

function PropagateBatch:__init(model_id, input_state, carry_state, batch_state)
   parent.init(model_id)
   self._input_state = input_state
   self._carry_state = carry_state
   self._batch_state = batch_state
end

function Propagate:execute(session)
   local session_map, shared_map = session:objectMaps()
   local model = getSharedWithLocalMemento(self._subject_id)
   local output_state, carry_state = model:forward(
      self._input_state, self._carry_state, self._batch_state
   )
   return output_state, carry_state
end
