------------------------------------------------------------------------
--[[ BaseSet ]]--
-- Base class inherited by DataSet and Batch.
------------------------------------------------------------------------
local BaseSet = torch.class("dp.BaseSet")
BaseSet.isBaseSet = true

function BaseSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, which_set, inputs, targets
      = xlua.unpack(
      {config},
      'BaseSet', 
      'Base class inherited by DataSet and Batch.',
      {arg='which_set', type='string',
       help='"train", "valid" or "test" set'},
      {arg='inputs', type='dp.BaseTensor | table of dp.BaseTensors', 
       help='Sample inputs to a model. These can be DataTensors or '..
       'a table of BaseTensors (in which case these are converted '..
       'to a CompositeTensor', req=true},
      {arg='targets', type='dp.BaseTensor | table of dp.BaseTensors', 
       help='Sample targets to a model. These can be BaseTensors or '..
       'a table of BaseTensors (in which case these are converted '..
       'to a CompositeTensor. The indices of examples must be '..
       'in both inputs and targets must be aligned.'}
   )
   self:setWhichSet(which_set)
   if inputs then self:setInputs(inputs) end
   if targets then self:setTargets(targets) end
end

function BaseSet:setWhichSet(which_set)
   self._which_set = which_set
end

function BaseSet:whichSet()
   return self._which_set
end

function BaseSet:isTrain()
   return (self._which_set == 'train')
end

function BaseSet:setInputs(inputs)
   if not torch.typename(inputs) and type(inputs) == 'table' then
      --if list, make CompositeTensor
      inputs = dp.CompositeTensor{components=inputs}
   end
   assert(inputs.isBaseTensor, 
      "Error : invalid inputs. Expecting type dp.BaseTensor")
   self._inputs = inputs
end

function BaseSet:setTargets(targets)
   if not torch.typename(targets) and type(targets) == 'table' then
      --if list, make CompositeTensor
      targets = dp.CompositeTensor{components=targets}
   end
   assert(targets.isBaseTensor,
      "Error : invalid targets. Expecting type dp.BaseTensor")
   self._targets = targets
end

-- Returns the number of samples in the BaseSet.
function BaseSet:nSample()
   return self._inputs:nSample()
end

--Returns input dp.DataTensors
function BaseSet:inputs()
   return self._inputs
end

--Returns target dp.DataTensors
function BaseSet:targets()
   return self._targets
end

--Preprocesses are applied to DataTensors, which means that 
--DataTensor:image(), :expandedAxes(), etc. can be used.
function BaseSet:preprocess(...)
   local args, input_preprocess, target_preprocess, can_fit
      = xlua.unpack(
         {... or {}},
         'BaseSet:preprocess',
         'Preprocesses the BaseSet.',
         {arg='input_preprocess', type='dp.Preprocess', 
          help='Preprocess applied to the input DataTensor(s) of ' .. 
          'the BaseSet'},
         {arg='target_preprocess', type='dp.Preprocess',
          help='Preprocess applied to the target DataTensor(s) of ' ..
          'the BaseSet'},
         {arg='can_fit', type='boolean',
          help='Allows measuring of statistics on the DataTensor(s) ' .. 
          'of BaseSet to initialize the preprocess. Should normally ' .. 
          'only be done on the training set. Default is to fit the ' ..
          'training set.'}
   )
   assert(input_preprocess or target_preprocess, 
      "Error: no preprocess (neither input nor target) provided)")
   if can_fit == nil then
      can_fit = self:isTrain()
   end
   --TODO support multi-input/target preprocessing
   if input_preprocess and input_preprocess.isPreprocess then
      input_preprocess:apply(self._inputs, can_fit)
   end
   if target_preprocess and target_preprocess.isPreprocess then
      target_preprocess:apply(self._targets, can_fit)
   end
end
