------------------------------------------------------------------------
--[[ PGDone ]]--
-- Observer
-- Maintains SQL table dp.done, which keeps a log of 
-- completed experiments.
-- Only works with experiments (could do without by using root id...)
------------------------------------------------------------------------

local PGDone, parent = torch.class("dp.PGDone", "dp.Observer")

function PGDone:__init(...)
   local args, pg = xlua.unpack(
      {... or {}},
      'PGDone', nil,
      {arg='pg', type='dp.Postgres',
       help='Postgres connection instance. Default is a dp.Postgres()'}
   )
   self._pg = pg or dp.Postgres()
   parent.__init(self, "finalizeExperiment")
end

function PGDone:setSubject(subject)
   assert(subject.isExperiment)
   self._subject = subject
end

function PGDone:finalizeExperiment()
   self._pg:execute(
      "INSERT INTO dp.done (xp_id, end_time) " ..
      "VALUES (%s, now())",
      {self._subject:name()}
   )
end

