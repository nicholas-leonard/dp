------------------------------------------------------------------------
--[[ PGQuery ]]--
-- Used to query information found in the postgresql logs
------------------------------------------------------------------------
local PGQuery = torch.class("PGQuery")

function PGQuery:__init()
   local args, pg = xlua.unpack(
      {config or {}},
      'PGQuery', nil,
      {arg='pg', type='dp.Postgres', default=dp.Postgres()}
   )
   parent.__init(self)
   self._pg = pg
end

function PGQuery:selectHyperparam(xp_id)
   assert(type(xp_id) == 'number')
   local row = self._pg:fetchOne(
      "SELECT hyperparam_pickle " .. 
      "FROM dp.hyperparam " ..
      "WHERE xp_id = %s", 
      {xp_id}
   )
   local hyperparams = row[1]
   if hyperparams then 
      return torch.deserialize(hyperparams)
   end
end

function PGQuery:selectReport(xp_id, epoch)
   assert(type(xp_id) == 'number')
   local report
   if epoch then
      local row = self._pg:fetchOne(
         "SELECT report_pickle " .. 
         "FROM dp.report " ..
         "WHERE (xp_id, report_epoch) = (%s, %s)", 
         {xp_id, epoch}
      )
      report = row[1]
      if report then
         return torch.deserialize(report)
      end
   else
      local rows = self._pg:fetch(
         "SELECT epoch, report_pickle " .. 
         "FROM dp.report " ..
         "WHERE xp_id = %s", 
         {xp_id}
      )
      if _.isEmpty(rows) then return end
      local reports = {}
      for i, row in ipairs(rows) do
         reports[row[1]] = torch.deserialize(row[2])
      end
      return reports
   end
end

function PGQuery:selectExperiment(xp_id)
   assert(type(xp_id) == 'number')
   local row = self._pg:fetchOne(
      "SELECT xp_id, collection_name, process_name, " ..
             "xp_factory_name, ds_factory_name " .. 
      "FROM dp.experiment" ..
      "WHERE xp_id = %s", 
      {xp_id},
      'a'
   )
   if not _.isEmpty(row) then
      return row
   end
end

function PGQuery:selectCollection(collection_name)
   assert(type(xp_id) == 'string')
   local row = self._pg:fetchOne(
      "SELECT xp_id, process_name, " ..
             "xp_factory_name, ds_factory_name " .. 
      "FROM dp.experiment" ..
      "WHERE xp_id = %s", 
      {xp_id},
      'a'
   )
   if not _.isEmpty(row) then
      return row
   end
end

function PGQuery:_experiment(xp_id)
   local xp = self:selectExperiment(xp_id)
   if not xp then return end
   xp.hyperparams = self:selectHyperparam(xp_id)
   xp.reports = self:selectReport(xp_id)
   return xp
end

--[[
function Experiment:__init(...)
   self._xp_id
   self._collection_name
   self._xp_factory_name
   self._ds_factory_name
   self._start_time
   self._end_time
end

function Experiment:refresh(...)

end

function Experiment:isDone()
   return self._end_time
end]]--

