require 'torch'
require 'image'
require 'paths'
require 'dok'
require 'xlua'

------------------------------------------------------------------------
-- TODO : 
--- Preprocessor.__init should have a switch to determine if it is
--- to be applied to inputs, targets or both. Default is inputs only.
--- Unit tests. Start with standardize
------------------------------------------------------------------------

--[[
Abstract class.

An object that can preprocess a dataset.

Preprocessing a dataset implies changing the data that
a dataset actually stores. This can be useful to save
memory--if you know you are always going to access only
the same processed version of the dataset, it is better
to process it once and discard the original.

Preprocessors are capable of modifying many aspects of
a dataset. For example, they can change the way that it
converts between different formats of data. They can
change the number of examples that a dataset stores.
In other words, preprocessors can do a lot more than
just example-wise transformations of the examples stored
in the dataset.

function Preprocess:__init(...)
   
end
]]--
local Preprocess = torch.class("dp.Preprocess")


--[[
datatensor: The DataTensor to act upon. An instance of dp.DataTensor.

can_fit: If True, the Preprocess can adapt internal parameters
         based on the contents of dataset.

Typical usage:
    # Learn PCA preprocessing and apply it to the training set
    train_set = MyDataset{which_set='train'}
    my_pca_preprocess:apply(train_set)
    # Now apply the same transformation to the test set
    test_set = MyDataset{which_set='valid'}
    my_pca_preprocess:apply(test_set)
]]--
function Preprocess:apply(datatensor, can_fit)
   error("Preprocessor subclass does not implement an apply method.")
end

Preprocess.isPreprocess = true

-----------------------------------------------------------------------
-- Pipeline : A Preprocessor that sequentially applies a list
-- of other Preprocessors.
-----------------------------------------------------------------------
local Pipeline = torch.class("dp.Pipeline", "dp.Preprocess")

function Pipeline:__init(items)
   self._items = items or {}
end

function Pipeline:apply(datatensor, can_fit)
   for i, item in ipairs(self._items) do
      item:apply(datatensor, can_fit)
   end
end
            
            

-----------------------------------------------------------------------
-- Binarize : A Preprocessor that set to 0 any pixel strictly below the 
---threshold, sets to 1 those above or equal to the threshold.
-----------------------------------------------------------------------
local Binarize = torch.class("dp.Binarize", "dp.Preprocess")

function Binarize:__init(threshold)
   self._threshold = threshold
end

function Binarize:apply(datatensor)
   local data = datatensor:data()
   inputs[data:lt(self._threshold)] = 0;
   inputs[data:ge(self._threshold)] = 1;
   datatensor:setData(data)
end

-----------------------------------------------------------------------
-- Standardize : A Preprocessor that subtracts the mean and divides 
-- by the standard deviation.
-----------------------------------------------------------------------
local Standardize = torch.class("dp.Standardize", "dp.Preprocess")

function Standardize:__init(...)
   local args
   args, self._global_mean, self._global_std, self._std_eps
      = xlua.unpack(
      {... or {}},
      'Standardize', nil,
      {arg='global_mean', type='boolean', 
       help=[[If true, subtract the (scalar) mean over every element
            in the dataset. If false, subtract the mean from
            each column (feature) separately.
            ]], default=false},
      {arg='global_std', type='boolean', 
       help=[[If true, after centering, divide by the (scalar) standard
            deviation of every element in the design matrix. If false,
            divide by the column-wise (per-feature) standard deviation.
            ]], default=false},
      {arg='std_eps', type='number', 
       help=[[Stabilization factor added to the standard deviations before
            dividing, to prevent standard deviations very close to zero
            from causing the feature values to blow up too much.
            ]], default=1e-4}
   )
end
    
function Standardize:apply(datatensor, can_fit)
   local data = datatensor:feature()
   if can_fit then
      self._mean = self._global_mean and data:mean() or data:mean(1)
      self._std = self._global_std and data:std() or data:std(1)
   elseif self._mean == nil or self._std == nil then
          error([[can_fit is false, but Standardize object
                  has no stored mean or standard deviation]])
   end
   if self._global_mean then
      data:add(-self._mean)
   else
      -- broadcast
      data:add(-self._mean:expandAs(data))
   end
   if self._global_std then
      data:cdiv(self._std + self._std_eps)
   else
      data:cdiv(self._std:expandAs(data) + self._std_eps)
   end
   datatensor:setData(data)
end
