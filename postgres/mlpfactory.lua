------------------------------------------------------------------------
--[[ PGMLPFactory ]]--
-- An example experiment builder for training Mnist using an 
-- MLP of arbitrary dept
------------------------------------------------------------------------
local PGMLPFactory, parent = torch.class("dp.PGMLPFactory", "dp.MLPFactory")
PGMLPFactory.isPGMLPFactory = true
   
function PGMLPFactory:__init(config)
   config = config or {}
   local args, name, pg = xlua.unpack(
      {config},
      'PGMLPFactory', nil,
      {arg='name', type='string', default='MLP', help=''},
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   config.name = name
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGMLPFactory:buildObserver(opt)
   return {
      self._logger,
      dp.PGEarlyStopper{
         start_epoch = 1,
         pg = self._pg,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy
      },
      dp.PGDone{pg=self._pg}
   }
end
