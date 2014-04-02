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

function GCN:__init(...)
   local args
   args, self._substract_mean, self._scale, self._sqrt_bias, 
   self._use_std, self._min_divisor, self._batch_size
      = xlua.unpack(
      {... or {}},
      'GCN', nil,
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
      {arg='batch_size', type='number', default=0, 
       help='The size of a batch.'}
   )
end
    
function GCN:apply(datatensor, can_fit)
   local data = datatensor:feature()
   print('begin GCN preprocessing...')
   if self._batch_size == 0 then
      self:_transform(data)
   else
      local data_size = data:size(1)
      local last = math.floor(data_size / self._batch_size) * self._batch_size

      for i = 0, data_size, self._batch_size do
         if i >= last then
            stop = i + math.mod(data_size, self._batch_size)
         else
            stop = i + self._batch_size
         end
         self:_transform(data:sub(i,stop))
      end
   end
   datatensor:setData(data)
   print('GCN preprocessing completed')
end

function GCN:_transform(data)
   local mean = data:mean(2)
   if self._substract_mean then
      data:add(mean:mul(-1):resize(mean:size(1),1):expandAs(data))
   end
	local scale
	if self._use_std then
		scale = torch.sqrt(torch.pow(data,2):mean(2):add(self._sqrt_bias))
	else
      scale = torch.sqrt(torch.pow(data,2):sum(2):add(self._sqrt_bias))
	end
	local eps = 1e-8
	scale[torch.lt(scale, eps)] = 1
	data:cdiv(scale:expandAs(data)):mul(self._scale)
end

