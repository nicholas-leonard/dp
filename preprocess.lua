require 'torch'
require 'image'
require 'paths'

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

-----------------------------------------------------------------------
-- GlobalContrastNormalization : Global contrast normalizes by (optionally) 
-- subtracting the mean across features and then normalizes by either the 
-- vector norm or the standard deviation (across features, for each example).
-----------------------------------------------------------------------
local GlobalContrastNormalize 
 = torch.class("dp.GlobalContrastNormalize", "dp.Preprocess")


function GlobalContrastNormalize:__init(...)
   local args
   args, self._subtract_mean, self._scale, self._sqrt_bias, self._use_std,
   self._min_divisor, self._std_bias, self._use_norm, self._batch_size
      = xlua.unpack(
      {...},
      'GlobalContrastNormalize', nil,
      {arg='subtract_mean', type='boolean', 
       help=[[if True subtract the mean of each example.
            ]], default=true},
            
      {arg='scale', type='number', 
       help=[[Multiply features by this const.
            ]], default=1.0},
            
      {arg='sqrt_bias', type='number', 
       help=[[Fudge factor added inside the square root.
            ]], default=0},
            
      {arg='use_std', type='boolean', 
       help=[[If True uses the norm instead of the standard deviation
            ]], default=false},
            
      {arg='min_divisor', type='number', 
       help=[[If the divisor for an example is less than this value,
        	do not apply it.
            ]], default=1e-8},
            
      {arg='std_bias', type='number', 
       help=[[Add this amount inside the square root when computing
            the standard deviation or the norm
            ]], default=0},

      {arg='batch_size', type='number', 
       help=[[The size of a batch.
            ]], default=0}
   )
end
    
function GlobalContrastNormalize:apply(datatensor, can_fit)
   local data = datatensor:feature()
   if self._batch_size == 0 then
      data = self:_transform(data)
      datatensor.setData(data)    
   else
      local data_size = data:size(1)
      local last = math.floor(data_size / self._batch_size) * self._batch_size

      for i = 0, data_size, self._batch_size do
         if i >= last then
            stop = i + math.mod(data_size, self._batch_size)
         else
            stop = i + self._batch_size
         end
      end

      data = self:_transform(data:sub(1,stop))
      datatensor.setData(data)
   end
end

function GlobalContrastNormalize:_transform(data)
	if self._subtract_mean then
		local miu = torch.mean(data,2)
		miu = torch.expand(miu, data:size())
		data = data - miu
	end
	
	local sqr = torch.pow(data,2)
	if self._use_norm then
		scale = torch.sqrt(torch.sum(sqr,2) + self._std_bias)
	else
		scale = torch.sqrt(torch.mean(sqr,2) + self.std_bias)
	end
	
	local eps = 1e-8
	scale[torch.lt(scale, eps)] = 1
	data = torch.cdiv(data, torch.expand(scale, data:size()))
	return data
end


-----------------------------------------------------------------------
-- ZCA : Performs ZCA Whitening
-----------------------------------------------------------------------

local ZCA = torch.class("dp.ZCA", "dp.Preprocess")


function ZCA:__init(...)
   local args
   args, self.n_components, self.n_drop_components, self.filter_bias, 
   self.copy, self.has_fit
      = xlua.unpack(
      {... or {}},
      'ZCA', nil,
      {arg='n_component', type='number', 
       help=[[
            ]], default=nil},
            
      {arg='n_drop_component', type='number', 
       help=[[
            ]], default=nil},
            
      {arg='filter_bias', type='number', 
       help=[[
            ]], default=0.1},
            
      {arg='copy', type='boolean', 
       help=[[
            ]], default=true},
            
      {arg='has_fit', type='boolean', 
       help=[[
            ]], default=false}
   )
end

function ZCA:fit(X)
    
   assert (X:dim() == 2)
   local n_samples = X:size()[0]
         
   -- center data
   self.mean_ = torch.mean(X, 0)
   X = X - self.mean_

   if self.copy then
     X = X:clone()
   end

   print('computing ZCA')
   local matrix = torch.mm(X:t(), X) / X:size()[0] + self.filter_bias * torch.eye(X:size()[1])
   local eig_val, eig_vec = torch.eig(matrix, 'V')
   eig_val = eig_val:sub(1,-1,1,1):reshape(eig_val:size()[1])
   local sorted_eig_val, sorted_index = eig_val:sort(1)
   local sorted_eig_vec = eig_vec:index(2, sorted_index)

   assert(eig_val:min() > 0)
   if self.n_components then
     sorted_eig_val = sorted_eig_val:sub(1, self.n_components)
     sorted_eig_vec = sorted_eig_vec:sub(1, -1, 1, self.n_components)
   end
   if self.n_drop_components then
      sorted_eig_val = sorted_eig_val:sub(self.n_drop_component, -1)
      sorted_eig_vec = sorted_eig_vec:sub(1, -1, self.n_drop_component, -1)
   end

   sorted_eig_val:apply(function(e) e = 1/e; return e; end)
   eig_vec_T = sorted_eig_vec:clone().t()
   local index = 0
   sorted_eig_vec:apply(function(e) 
                           e = e * sorted_eig_val[index+1];
                           index = (index + 1) % sorted_eig_vec:size(2);
                           return e;
                        end) 

   self.P_ = torch.mm(sorted_eig_vec, eig_vec_T)
   self.has_fit_ = true
end

function ZCA:apply(datatensor, can_fit)
   local X = datatensor:feature()
   if self.has_fit_ == false then
      self:fit(X)
   end
   new_X = torch.mm(X - self.mean_, self.P_)
   datatensor.setData(new_X)
end
