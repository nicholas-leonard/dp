------------------------------------------------------------------------
--[[ ImageClassFactory ]]--
-- interface, factory
-- Builds Image Classification datasources with common preprocessings
------------------------------------------------------------------------
local ImageClassFactory, parent = torch.class("dp.ImageClassFactory", "dp.DatasourceFactory")
ImageClassFactory.isImageClassFactory = true

function ImageClassFactory:__init(...)
   parent.__init(self, {name='ImageClass'})
end

function ImageClassFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource))
   else
      datasource = dp[opt.datasource]{
         valid_ratio=opt.valid_ratio, 
         input_preprocess=self:buildInputPreprocess(opt)
      }
      self._cache[opt.datasource] = datasource
   end
   opt.feature_size = datasource._feature_size
   opt.classes = datasource._classes
   return datasource
end

function ImageClassFactory:buildInputPreprocess(opt)
   local input_preprocess
   if opt.zca_gcn then
      input_preprocess = {dp.GCN(),dp.ZCA()} --order is important
   elseif opt.standardize then
      input_preprocess = dp.Standardize()
   elseif opt.lecunlcn then
      input_preprocess = dp.LeCunLCN()
   end
   return input_preprocess
end
