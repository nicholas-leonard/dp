
------------------------------------------------------------------------
--[[ Query ]]--
-- interface
-- Used to query information found in the logs
------------------------------------------------------------------------
local Query = torch.class("Query")
Query.isQuery = true

function Query:__init()
   self._cache = {experiment = {}, collection = {}}
end

function Query:experiment(xp_id)
   local experiment = self._cache.experiment[xp_id] 
   if not experiment then 
      experiment = self:_experiment(xp_id)
      self._cache.experiment[xp_id] = experiment
   end
   if not experiment then
      print"Query:experiment() Warning : experiment not found"
   end
   return experiment
end

function Query:collection(collection_name)
   local collection = self._cache.collection[collection_name] 
   if not collection then 
      collection = self:_collection(collection_name)
      self._cache.collection[collection_name] = collection
   end
   if not experiment then
      print"Query:collection() Warning : collection not found"
   end
   return collection
end

