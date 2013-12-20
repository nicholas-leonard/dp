------------------------------------------------------------------------
--[[ PGEIDGenerator ]]--
-- EIDGenerator
-- Generates a unique identifier for the experiment using a postgresql
-- sequence.
------------------------------------------------------------------------

local PGEIDGenerator, parent 
   = torch.class("dp.PGEIDGenerator", "dp.EIDGenerator")
PGEIDGenerator.isPGEIDGenerator = true

function PGEIDGenerator:__init(config)
   local args, pg, sequence = xlua.unpack(
      {config or {}},
      'PGEIDGenerator', 'Generates unique IDs for experiments using ' ..
      'a postgreSQL Sequence.',
      {arg='pg', type='dp.Postgres', default=dp.Postgres()},
      {arg='sequence', type='string', default='dp.xp_id_gen'}
   )
   self._pg = pg
   self._sequence = sequence
end

function PGEIDGenerator:nextID()
   local eid = self._pg:fetchOne("SELECT nextval('%s')", 
                                 {self._sequence})[1]
   return dp.ObjectID(eid)
end

