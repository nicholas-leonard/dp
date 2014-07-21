------------------------------------------------------------------------
--[[ BillionWordsFactory ]]--
-- interface, factory
-- Builds BillionWords datasource instances
------------------------------------------------------------------------
local BillionWordsFactory, parent = torch.class("dp.BillionWordsFactory", "dp.DatasourceFactory")
BillionWordsFactory.isBillionWordsFactory = true

function BillionWordsFactory:__init(...)
   parent.__init(self, {name='BillionWords'})
end

function BillionWordsFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isBillionWords)
   elseif opt.datasource == 'billionwords' then
      local train_file = 'train_data.th7' 
      if opt.small then 
         train_file = 'train_small.th7'
      elseif opt.tiny then 
         train_file = 'train_tiny.th7'
      end
      local datasource = dp.BillionWords{
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
