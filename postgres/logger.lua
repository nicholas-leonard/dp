------------------------------------------------------------------------
--[[ PGLogger ]]--
-- Observer
-- Logger that logs a serialized report every epoch in a postgresql
-- database management system. Each epoch report is stored as a row in 
-- a table. Logger can only be used with PGEIDGenerator.

-- Note: if all sub-objects have a hierarchical unique id, they can be 
-- serialized
------------------------------------------------------------------------
local PGLogger, parent = torch.class("dp.PGLogger")
function PGLogger:__init(config)
   local args, pg = xlua.unpack(
      {config or {}},
      'PGLogger', nil,
      {arg='pg', type='dp.Postgres', default=dp.Postgres()}
   )
   parent.__init(self)
   self._pg = pg
end

--Actually this doesn't matter since report is sent by experiment
--[[function PGLogger:setSubject(subject)
   assert(subject.isExperiment)
   self._subject = subject
end]]--

function PGLogger:doneEpoch(report)
   local report_str = torch.serialize(report)
   self._pg:execute(
      "INSERT INTO dp.report (xp_id, epoch, report) " .. 
      "VALUES (%s, %s, '%s')", 
      {report.id, report.epoch, report_str}
   )
end
