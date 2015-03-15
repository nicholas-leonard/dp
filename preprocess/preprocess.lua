------------------------------------------------------------------------
--[[ Preprocess ]]--
--Abstract class.
--An object that can preprocess a View.
--Preprocessing a View implies changing the data that
--a View actually stores. This can be useful to save
--memory. If you know you are always going to access only
--the same processed version of the dataset, it is better
--to process it once and discard the original.

--Preprocesses are capable of modifying many aspects of
--a dataset. For example, they can change the way that it
--converts between different formats of data. They can
--change the number of examples that a dataset stores.
--In other words, preprocesses can do a lot more than
--just example-wise transformations of the examples stored
--in a basetensor.
------------------------------------------------------------------------
local Preprocess = torch.class("dp.Preprocess")
Preprocess.isPreprocess = true

-- view: The DataView to act upon. An instance of dp.DataView.
-- can_fit: If True, the Preprocess can adapt internal parameters
--    based on the contents of dataset.
function Preprocess:apply(view, can_fit)
   error("Preprocess subclass does not implement an apply method.")
end 
