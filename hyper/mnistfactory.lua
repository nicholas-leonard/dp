------------------------------------------------------------------------
--[[ MnistFactory ]]--
-- interface, factory
-- Builds Mnist datasource instances with common preprocessings
------------------------------------------------------------------------
local MnistFactory, parent = torch.class("dp.MnistFactory", "dp.DatasourceFactory")
MnistFactory.isMnistFactory = true

function MnistFactory:__init(...)
   parent.__init(self, {name='Mnist'})
end

function MnistFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isMnist)
   elseif opt.datasource == 'mnist' then
      datasource = dp.Mnist()
      self._cache[opt.datasource] = datasource
   elseif opt.datasource == 'mnist:standardize' then
      datasource = dp.Mnist{input_preprocess = dp.Standardize}
      self._cache[opt.datasource] = datasource
   else
      error("Unknown datasource : " .. opt.datasource)
   end
   -- to be used by experiment builder
   opt.feature_size = datasource._feature_size
   opt.classes = datasource._classes
   return datasource
end
