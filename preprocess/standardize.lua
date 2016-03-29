-----------------------------------------------------------------------
--[[ Standardize ]]--
-- A Preprocess that subtracts the mean and divides 
-- by the standard deviation.
-----------------------------------------------------------------------
local Standardize = torch.class("dp.Standardize", "dp.Preprocess")
Standardize.isStandardize = true

function Standardize:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args
   args, self._global_mean, self._global_std, self._std_eps
      = xlua.unpack(
      {config},
      'Standardize', nil,
      {arg='global_mean', type='boolean', default=false,
       help='If true, subtract the (scalar) mean over every element '..
       'in the dataset. If false, subtract the mean from each column '..
       '(feature) separately.'},
      {arg='global_std', type='boolean', default=false,
       help='If true, after centering, divide by the (scalar) '..
       'standard deviation of every element in the design matrix. '..
       'If false, divide by the column-wise (per-feature) standard '..
       'deviation.'},
      {arg='std_eps', type='number', default=1e-4,
       help='Stabilization factor added to the standard deviations '..
       'before dividing, to prevent standard deviations very close to '..
       'zero from causing the feature values to blow up too much.'}
   )
end

function Standardize:apply(dv, can_fit)
   assert(dv.isDataView, "Expecting a DataView instance")
   local data = dv:forward('bf')
   if can_fit then
      self._mean = self._global_mean and data:mean() or data:mean(1)
      self._std = self._global_std and data:std() or data:std(1)
   elseif self._mean == nil or self._std == nil then
      error("can_fit is false, but Standardize object "..
            "has no stored mean or standard deviation")
   end
   if self._global_mean then
      data:add(-self._mean)
   else
      -- broadcast
      data:add(-self._mean:expandAs(data))
   end
   if self._global_std then
      data:div(self._std + self._std_eps)
   else
      data:cdiv(self._std:expandAs(data) + self._std_eps)
   end
   dv:replace('bf', data)
end
