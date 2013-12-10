------------------------------------------------------------------------
--[[ EIDGenerator ]]--
-- Generates a unique identifier for the experiment.
-- Default is to concatenate a provided namespace and 
-- the time of the experiment, and the next value from a sequence
-- as a unique name
-- To ensure uniqueness across experiments, the namespace should 
-- be associated to the process, and there should be but one 
-- EIDGenerator instance per process.

-- Like Builder, Mediator and Data*, this object exists in the 
-- extra-experiment scope.
------------------------------------------------------------------------

local EIDGenerator = torch.class("dp.EIDGenerator")
EIDGenerator.isEIDGenerator = true

function EIDGenerator:__init(namespace, seperator)
   self._namespace = namespace
   self._index = 0
   self._seperator = seperator or '.'
end

function EIDGenerator:nextID()
   local eid = self._namespace .. os.time() .. 
               self._seperator .. self._index
   self._index = self._index + 1
   return dp.ObjectID(eid)
end
