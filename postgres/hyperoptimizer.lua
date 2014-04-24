------------------------------------------------------------------------
--[[ HyperOptimizer ]]--
-- Samples hyperparameters from a HyperparamSampler and 
-- passes these to a DatasourceFactory and ExperimentFactory to 
-- generate an Experiment wich is run until it terminates (returns) and
-- then the cyle starts again.
------------------------------------------------------------------------
local PGHyperOptimizer, parent = torch.class("dp.PGHyperOptimizer", "dp.HyperOptimizer")
PGHyperOptimizer.isPGHyperOptimizer = true

function PGHyperOptimizer:__init(config)
   config = config or {}
   local args, pg, sequence = xlua.unpack(
      {config or {}},
      'HyperOptimizer', nil,
      {arg='pg', type='dp.Postgres', help='defaults to dp.Postgres()'},
      {arg='sequence', type='string', default='dp.xp_id_gen',
       help='name of the SQL Sequence to use for generating unique ids'}
   )
   self._pg = pg or dp.Postgres()
   assert(type(sequence) == 'string')
   self._sequence = sequence
   parent.__init(self, config)
end

-- Generates a unique identifier for the experiment using an SQL sequence
function PGHyperOptimizer:nextID()
   local id = self._pg:fetchOne("SELECT nextval('%s')", {self._sequence})[1]
   return dp.ObjectID(id)
end
