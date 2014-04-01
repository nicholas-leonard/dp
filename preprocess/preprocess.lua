------------------------------------------------------------------------
--[[ Preprocess ]]--
--Abstract class.
--An object that can preprocess a dataset.
--Preprocessing a dataset implies changing the data that
--a dataset actually stores. This can be useful to save
--memory--if you know you are always going to access only
--the same processed version of the dataset, it is better
--to process it once and discard the original.

--Preprocessors are capable of modifying many aspects of
--a dataset. For example, they can change the way that it
--converts between different formats of data. They can
--change the number of examples that a dataset stores.
--In other words, preprocessors can do a lot more than
--just example-wise transformations of the examples stored
--in the dataset.
------------------------------------------------------------------------
local Preprocess = torch.class("dp.Preprocess")
Preprocess.isPreprocess = true

function Preprocess:__init(...)
end

--datatensor: The DataTensor to act upon. An instance of dp.DataTensor.
--can_fit: If True, the Preprocess can adapt internal parameters
--    based on the contents of dataset.
--Typical usage:
--    # Learn PCA preprocessing and apply it to the training set
--    train_set = MyDataset{which_set='train'}
--    my_pca_preprocess:apply(train_set)
--    # Now apply the same transformation to the test set
--    test_set = MyDataset{which_set='valid'}
--    my_pca_preprocess:apply(test_set)
function Preprocess:apply(datatensor, can_fit)
   error("Preprocessor subclass does not implement an apply method.")
end 
