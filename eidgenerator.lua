------------------------------------------------------------------------
--[[ EIDGenerator ]]--
-- Generates a unique identifier for the experiment.
-- If a namespace is provided it is concatenated with 
-- the time of the experiment, and the next value from a sequence
-- to get a pseudo-unique name.
-- To ensure uniqueness across experiments, the namespace should 
-- be associated to the process, and their should be but one 
-- EIDGenerator instance per process.

-- When no namespace is provided, we concatenate the linux hostname 
-- and PID.

-- Like HyperOptimizer, Mediator and Data*, this object exists in the 
-- extra-experiment scope.
------------------------------------------------------------------------
local EIDGenerator = torch.class("dp.EIDGenerator")
EIDGenerator.isEIDGenerator = true

function EIDGenerator:__init(namespace, separator)
   self._separator = separator or ':'
   self._namespace = namespace or os.hostname()..self._separator..os.pid()
   self._index = 0
end

function EIDGenerator:nextID()
   local eid = self._namespace..self.separator..os.time()..self._separator..self._index
   self._index = self._index + 1
   return dp.ObjectID(eid)
end

local counter = 1
function dp.uniqueID(namespace, separator)
   local separator = separator or ':'
   local namespace = namespace or os.hostname()..separator..os.pid()
   local uid = namespace..separator..os.time()..separator..counter
   counter = counter + 1
   return uid
end

