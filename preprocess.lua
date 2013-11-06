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
local Preprocess = torch.class("data.Preprocess")


--[[
dataset: The dataset to act on. An instance of data.Dataset.

can_fit: If True, the Preprocess can adapt internal parameters
         based on the contents of dataset. Default is to fit 
         the preprocessor to dataset:isTrain() set, and to reuse 
         that fitting for the dataset:isTrain() == false.

Typical usage:
    # Learn PCA preprocessing and apply it to the training set
    train_set = MyDataset{which_set='train'}
    my_pca_preprocess:apply(train_set)
    # Now apply the same transformation to the test set
    test_set = MyDataset{which_set='valid'}
    my_pca_preprocess:apply(test_set)
]]--
function Preprocess:apply(dataset, can_fit)
   error("Preprocessor subclass does not implement an apply method.")
end


-----------------------------------------------------------------------
-- Pipeline : A Preprocessor that sequentially applies a list
-- of other Preprocessors.
-----------------------------------------------------------------------
local Pipeline = torch.class("data.Pipeline", "data.Preprocess")

function Pipeline:__init(items)
   self.items = items or {}
end

function Pipeline:apply(dataset, can_fit)
   for item in self.items:
      item.apply(dataset, can_fit)
end
            
            

-----------------------------------------------------------------------
-- Binarize : A Preprocessor that set to 0 any pixel strictly below the 
---threshold, sets to 1 those above or equal to the threshold.
-----------------------------------------------------------------------
local Binarize = torch.class("data.Binarize", "data.Preprocess")

function Binarize:__init(threshold)
   self.threshold = threshold
end

function Binarize:apply(dataset)
   local inputs = dataset:inputs()
   inputs[inputs:lt(threshold)] = 0;
   inputs[inputs:ge(threshold)] = 1;
   dataset:setInputs(inputs)
end

-----------------------------------------------------------------------
-- Standardize : A Preprocessor that subtracts the mean and divides 
-- by the standard deviation.
-----------------------------------------------------------------------
local Standardize = torch.class("data.Standardize", "data.Preprocess")

function Standardize:__init(...)
   local args
   args, self.global_mean, self.global_std, self.std_eps
      = xlua.unpack(
      {...},
      'Standardize constructor', nil,
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
            ]], default=1e-4},
   )
   self.mean
   self.std
end
    
function Standardize:apply(dataset, can_fit)
   local inputs = dataset:inputs()
   if can_fit == nil then
      can_fit == dataset:isTrain()
   end
   if can_fit then
      self.mean = self.global_mean and inputs:mean() or inputs:mean(1)
      self.std = self.global_std and inputs:std() or inputs:std(1)
   elseif self.mean == nil or self.std == nil then
          error([[can_fit is false, but Standardize object
                  has no stored mean or standard deviation]])
   end
   inputs:add(-self.mean):div(self.std + self.std_eps)
   dataset:setInputs(inputs)
end
