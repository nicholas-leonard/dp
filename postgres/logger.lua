------------------------------------------------------------------------
--[[ PGLogger ]]--
-- Observer
-- Logger that logs a serialized report every epoch in a postgresql
-- database management system. Each epoch report is stored as a row in 
-- a table. Logger can only be used with PGEIDGenerator.
------------------------------------------------------------------------
local PGLogger, parent = torch.class("dp.PGLogger", "dp.Logger")

function PGLogger:__init(config)
   local args, pg = xlua.unpack(
      {config or {}},
      'PGLogger', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   parent.__init(self)
   self._pg = pg or dp.Postgres()
end

function PGLogger:doneEpoch(report)
   local report_str = torch.serialize(report, 'ascii')
   report_str = string.sub(report_str, 1, #report_str-1)
   self._pg:execute(
      "INSERT INTO dp.report " ..
      "(xp_id, report_epoch, report_pickle, report_time) " .. 
      "VALUES (%s, %s, '%s', now())", 
      {self._subject:name(), report.epoch, report_str}
   )
end

function PGLogger:logHyperReport(hr)
   local hr_pickle = torch.serialize(hr, 'ascii')
   --hr_pickle = string.gsub(hr_pickle, [[%c]], [[\%0]])
   hr_pickle = string.sub(hr_pickle, 1, #hr_pickle-1)
   self._pg:execute([[
      INSERT INTO dp.experiment 
      (xp_id, collection_name, process_name, hyper_report_pickle) 
      VALUES (%s, '%s', '%s', '%s')]],
      {hr.experiment_id, hr.collection_name, hr.process_name, hr_pickle}
   )
end
