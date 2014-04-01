-----------------------------------------------------------------------
--[[ GCN ]]-- 
-- Global Contrast Normalizes by (optionally) 
-- subtracting the mean across features and then normalizes by either 
-- the vector norm or the standard deviation (across features, for 
-- each example).
-----------------------------------------------------------------------
local GCN, parent = torch.class("dp.GCN", "dp.Preprocess")
GCN.isGCN = true

function GCN:__init(...)
   local args
   args, self._scale, self._sqrt_bias, self._use_std, self._min_divisor, 
   self._std_bias, self._batch_size, self._use_norm
      = xlua.unpack(
      {... or {}},
      'GCN', nil,
      {arg='scale', type='number', default=1.0, 
       help='Multiply features by this const.'},
      {arg='sqrt_bias', type='number', default=0,
       help='Fudge factor added inside the square root.'},
      {arg='use_std', type='boolean', default=false, 
       help='If True uses the norm instead of the standard deviation.'},
      {arg='min_divisor', type='number', default=1e-8,
       help='If the divisor for an example is less than this value, '..
       'do not apply it.'},
      {arg='std_bias', type='number', default=0,
       help='Add this amount inside the square root when computing '..
       'the standard deviation or the norm'},
      {arg='batch_size', type='number', default=0, 
       help='The size of a batch.'},
      {arg='use_norm', type='boolean', default=false,
       help='Normalize the data'}
   )
end
    
function GCN:apply(datatensor, can_fit)
   local data = datatensor:feature()
   print('begin Global Contrast Normalization Preprocessing...')
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
   print('Global Contrast Normalization preprocessing completed')
end

function GCN:_transform(data)
	local scale
	if self._use_norm then
		scale = torch.sqrt(data:pow(2):sum(2):add(self._std_bias))
	else
		scale = torch.sqrt(data:pow(2):mean(2):add(self._std_bias))
	end
	
	local eps = 1e-8
	scale[torch.lt(scale, eps)] = 1
	data:cdiv(scale:expandAs(data)):mul(self._scale)
end
