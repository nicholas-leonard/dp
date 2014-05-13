------------------------------------------------------------------------
--[[ SequenceTensor ]]-- 
-- A DataTensor holding a tensor of sequences.
-- Has axes = {'b','s','f'}. 
-- Output of nn.LookupTable, input of nn.Temporal*
------------------------------------------------------------------------
local SequenceTensor, parent = torch.class("dp.SequenceTensor", "dp.DataTensor")
SequenceTensor.isSequenceTensor = true

function SequenceTensor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data, axes, sizes
      = xlua.unpack(
      {config},
      'SequenceTensor', 
      'Builds a SequenceTensor out of torch.Tensor data. ',
      {arg='data', type='torch.Tensor', 
       help='A torch.Tensor with 2 dimensions or more.', req=true},
      {arg='axes', type='table', 
       help='A table defining the order and nature of each dimension.'..
       ' [Default={"b", "s", "f"}].'},
      {arg='sizes', type='table | torch.LongTensor', 
       help='A table or torch.LongTensor holding the sizes of the '.. 
       'commensurate dimensions in axes. This should be supplied '..
       'if the dimensions of the data is different from the number '..
       'of elements in the axes table, in which case it will be used '..
       'to : data:reshape(sizes). Default is data:size().'}
   )   
   axes = axes or {'b','s','f'}
   assert(#axes > 2, "Provide at least 2 axes for sequences")
   parent.__init(self, {data=data, axes=axes, sizes=sizes})
end

-- default view used for sequence preprocessing
function SequenceTensor:sequence(tensortype, inplace, contiguous)
   return self:sequenceBSF(tensortype, inplace, contiguous)
end

function SequenceTensor:sequenceBSF(tensortype, inplace, contiguous)
   local desired_axes = {'b','s','f'}
   local current_axes = self:storedAxes()
   local current_size = self:storedSize()
   local expanded_axes = self:expandedAxes()
   local expanded_size = self:expandedSize()
   assert((expanded_size:size(1) == 3), "No image size provided at construction")
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if not (data:dim() == 3 and table.eq(current_axes, desired_axes)) then
      expanded_axes, expanded_size 
         = parent.transpose('b', 1, expanded_axes, expanded_size)
      --resize to sequence size
      if data:dim() == 2 and _.contains(current_axes, 'f') then
         data = self:feature(nil, false, false)
         data:resize(expanded_size:storage())      
      elseif data:dim() == 3 then
         current_axes, current_size, data
            = parent.transpose('b', 1, current_axes, current_size, data)
      end
      --convert between image representations
      expanded_axes, expanded_size, data 
         = parent.transpose('f', 3, expanded_axes, expanded_size, data)
      expanded_axes, expanded_size, data 
         = parent.transpose('s', 2, expanded_axes, expanded_size, data)
      assert(table.eq(expanded_axes, desired_axes),
            "Error: unsupported conversion of axes formats")
   end
   data = self:_format(data, tensortype, contiguous)
   self:_store(data, desired_axes, inplace)
   return data
end


function SequenceTensor:conv1D(tensortype, inplace, contiguous)
   -- works with TemporalConvolution
   return self:sequenceBSF(tensortype, inplace, contiguous)
end

function SequenceTensor:default(tensortype, inplace, contiguous)
   return self:sequenceBSF(tensortype, inplace, contiguous)
end

function SequenceTensor:expand(tensortype, inplace, contiguous, axes)
   axes = axes or self._axes
   if table.eq(axes, {'b','s','f'}) then
      return self:sequenceBSF(tensortype, inplace, contiguous)
   else
      error"Unsupported expansion"
   end
end
      


