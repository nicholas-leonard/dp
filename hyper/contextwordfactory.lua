------------------------------------------------------------------------
--[[ ContextWordFactory ]]--
-- interface, factory
-- Builds context word datasources like BillionWords, where the 
-- task involves predicting the next work given a preceding 
-- context of words
------------------------------------------------------------------------
local ContextWordFactory, parent = torch.class("dp.ContextWordFactory", "dp.DatasourceFactory")
ContextWordFactory.isContextWordFactory = true

function ContextWordFactory:__init(...)
   parent.__init(self, {name='ContextWorkFactory'})
end

function ContextWordFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isBillionWords)
   elseif opt.datasource == 'BillionWords' then
      local train_file = 'train_data.th7' 
      if opt.small then 
         train_file = 'train_small.th7'
      elseif opt.tiny then 
         train_file = 'train_tiny.th7'
      end
      datasource = dp.BillionWords{
         context_size = opt.context_size, train_file = train_file
      }
      self._cache[opt.datasource] = datasource
   else
      error("Unknown datasource : " .. opt.datasource)
   end
   -- to be used by experiment builder
   opt.vocabularySize = datasource:vocabularySize()
   opt.nClasses = datasource:vocabularySize()
   return datasource
end
