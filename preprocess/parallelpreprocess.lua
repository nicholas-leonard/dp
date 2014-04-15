------------------------------------------------------------------------
--[[ ParallelPreprocess ]]--
-- Preprocess subclass
-- Composite of Preprocesses
-- Used for preprocessing CompositeTensors
------------------------------------------------------------------------
local ParallelPreprocess = torch.class("ParallelPreprocess", "dp.Preprocess")
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

function ParallelPreprocess:apply(compositetensor, can_fit)
   assert(compositetensor.isCompositeTensor, 
      "ParallelPreprocess error : expecting CompositeTensor")
   local nPP = table.length(self._items)
   local nDT =table.length(compositetensor:data())
   assert(nPP == nDT, "ParallelPreprocess error : Unequal amount of "..
      "elements in preprocess vs datatensor : "..nPP.." ~= "..nDT..","..
      "respectively.")
   for k, preprocess in pairs(self._items) do
      preprocess:apply(compositetensor:data()[k], can_fit)
   end
end 
