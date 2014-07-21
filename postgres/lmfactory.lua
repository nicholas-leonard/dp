------------------------------------------------------------------------
--[[ PGLMFactory ]]--
-- An example experiment builder for training BillionWords using a
-- Language Model of arbitrary dept
------------------------------------------------------------------------
local PGLMFactory, parent = torch.class("dp.PGLMFactory", "dp.LMFactory")
PGLMFactory.isPGLMFactory = true
   
function PGLMFactory:__init(config)
   config = config or {}
   local args, name, pg = xlua.unpack(
      {config},
      'PGLMFactory', nil,
      {arg='name', type='string', default='LM', help=''},
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   config.name = name
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGLMFactory:buildObserver(opt)
   return {
      self._logger,
      dp.PGEarlyStopper{
         start_epoch = 11,
         pg = self._pg,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10
      },
      dp.PGDone{pg=self._pg}
   }
end
