-----------------------------------------------------------------------
--[[ GCN ]]-- 
-- Global Contrast Normalizes by (optionally) subtracting the 
-- mean across features and normalizing by either 
-- the vector norm or the standard deviation (across features, for 
-- each example).
-- Notes
-- sqrt_bias = 10 and use_std = true (and defaults for all other
-- parameters) corresponds to the preprocessing used in :
-- A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
-- Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
-- http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
-----------------------------------------------------------------------
local GCN, parent = torch.class("dp.GCN", "dp.Preprocess")
GCN.isGCN = true

function GCN:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args
   args, self._substract_mean, self._scale, self._sqrt_bias, 
   self._use_std, self._min_divisor, self._batch_size, self._progress
      = xlua.unpack(
      {config},
      'GCN', 
      'Global Contrast Normalization',
      {arg='substract_mean', type='boolean', default=true,
       help='Remove the mean across features/pixels before '..
       'normalizing. Note that this is the per-example mean across '..
       'pixels, not the per-pixel mean across examples.'},
      {arg='scale', type='number', default=1.0, 
       help='Multiply features by this const.'},
      {arg='sqrt_bias', type='number', default=0,
       help='Fudge factor added inside the square root.'..
       'Add this amount inside the square root when computing '..
       'the standard deviation or the norm'},
      {arg='use_std', type='boolean', default=false, 
       help='If True uses the standard deviation instead of the norm.'},
      {arg='min_divisor', type='number', default=1e-8,
       help='If the divisor for an example is less than this value, '..
       'do not apply it.'},
      {arg='batch_size', type='number', default=256, 
       help='The size of a batch.'},
      {arg='progress', type='boolean', default=true, 
       help='display progress bar'}
   )
   -- buffers
   self._square = torch.Tensor()
   self._buffer = torch.Tensor()
   self._result = torch.Tensor()
end
    
function GCN:apply(dv, can_fit)   
   if self._progress then
      print"applying GCN preprocessing"
   end
   for stop, view in dv:ipairsSub(self._batch_size, true, true) do
      -- transform and replace original tensor
      view:replace("bf", 
         self:transform(
            view:forward("bf")
         ), true
      )
      
      if self._progress then
         -- display progress
         xlua.progress(stop, dv:nSample())
      end
   end
   
   -- for aesthetics :
   if self._progress then
      xlua.progress(dv:nSample(), dv:nSample())
   end
end

function GCN:transform(x)
   if torch.type(x) ~= torch.type(self._buffer) then
      self._buffer = self._buffer:type(torch.type(x))
      self._square = self._square:type(torch.type(x))
      self._result = self._result:type(torch.type(x))
   end

   self._buffer:mean(x,2)
   self._result:resizeAs(x):copy(x)
   if self._substract_mean then
      self._result:add(self._buffer:mul(-1):view(self._buffer:size(1),1):expandAs(x))
   end

   self._square:pow(self._result,2)
   if self._use_std then
      self._buffer:mean(self._square,2)
   else
      self._buffer:sum(self._square,2)
   end
   self._buffer:add(self._sqrt_bias):sqrt()

   self._buffer[torch.lt(self._buffer, 1e-8)] = 1
   self._result:cdiv(self._buffer:expandAs(x))
   self._result:mul(self._scale)
   return self._result
end

