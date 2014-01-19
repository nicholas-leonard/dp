------------------------------------------------------------------------
--[[ ImageClassFactory ]]--
-- interface, factory
-- Builds  datasource instances with common preprocessings
------------------------------------------------------------------------
local Cifar10Factory, parent = torch.class("dp.Cifar10Factory", "dp.DatasourceFactory")
Cifar10Factory.isCifar10Factory = true

function Cifar10Factory:__init(...)
   parent.__init(self, {name='Cifar10'})
end

function Cifar10Factory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isCifar10)
   elseif opt.datasource == 'cifar10' then
      datasource = dp.Cifar10{valid_ratio=opt.valid_ratio}
      self._cache[opt.datasource] = datasource
   elseif opt.datasource == 'cifar10:zca:gcn' then
      datasource = dp.Cifar10{
         input_preprocess={dp.ZCA(),dp.GCN()}, valid_ratio=opt.valid_ratio
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
