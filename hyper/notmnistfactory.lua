------------------------------------------------------------------------
--[[ NotMnistFactory ]]--
-- interface, factory
-- Builds NotMnist datasource instances with common preprocessings
------------------------------------------------------------------------
local NotMnistFactory, parent = torch.class("dp.NotMnistFactory", "dp.DatasourceFactory")
NotMnistFactory.isNotMnistFactory = true

function NotMnistFactory:__init(...)
   parent.__init(self, {name='NotMnist'})
end

function NotMnistFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isNotMnist)
   elseif opt.datasource == 'notmnist' then
      datasource = dp.NotMnist{valid_ratio=opt.valid_ratio}
      self._cache[opt.datasource] = datasource
   elseif opt.datasource == 'notmnist:standardize' then
      datasource = dp.NotMnist{
         input_preprocess=dp.Standardize, valid_ratio=opt.valid_ratio
      }
      self._cache[opt.datasource] = datasource
   else
      error("Unknown datasource : " .. opt.datasource)
   end
   -- to be used by experiment builder
   opt.feature_size = datasource._feature_size
   opt.classes = datasource._classes
   return datasource
end
