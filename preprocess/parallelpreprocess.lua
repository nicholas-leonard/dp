------------------------------------------------------------------------
--[[ ParallelPreprocess ]]--
-- Preprocess subclass
-- Composite of Preprocesses
-- Used for preprocessing ListViews
------------------------------------------------------------------------
local ParallelPreprocess = torch.class("dp.ParallelPreprocess", "dp.Preprocess")
ParallelPreprocess.isParallelPreprocess = true

function ParallelPreprocess:__init(items)
   assert(type(items) == 'table', 
      "ParallelPreprocess requires a table of preprocesses")
   -- assert that items are preprocesses
   for k, preprocess in pairs(items) do
      assert(preprocess.isPreprocess, 
         "ParallelPreprocess error : expecting preprocess at index "..k)
   end
   self._items = items
end

function ParallelPreprocess:apply(listview, can_fit)
   assert(ListView.isListView, 
      "ParallelPreprocess error : expecting ListView")
   local nPP = table.length(self._items)
   local nDV = table.length(listview:components())
   assert(nPP == nDV, "ParallelPreprocess error : Unequal amount of "..
      "elements in preprocess vs listview : "..nPP.." ~= "..nDT..","..
      "respectively.")
   for k, preprocess in pairs(self._items) do
      preprocess:apply(listview:components()[k], can_fit)
   end
end 
