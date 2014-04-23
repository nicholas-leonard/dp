--useful for validating if an object is an instance of a class, 
--even when the class is a super class.
--e.g pattern = "^torch[.]%a*Tensor$"
--typepattern(torch.Tensor(4), pattern) and
--typepattern(torch.DoubleTensor(5), pattern) are both true.
--typepattern(3, pattern)
function typepattern(obj, pattern)
   local class = type(obj)
   if class == 'userdata' then
      class = torch.typename(obj)
   end
   local match = string.match(class, pattern)
   if match == nil then
      match = false
   end
   return match
end

function torch.isTensor(obj)
   return typepattern(obj, "^torch[.]%a*Tensor$")
end

function typeString_to_tensorType(type_string)
   if type_string == 'cuda' then
      return 'torch.CudaTensor'
   elseif type_string == 'float' then
      return 'torch.FloatTensor'
   elseif type_string == 'double' then
      return 'torch.DoubleTensor'
   end
end

-- warning : doesn't allow non-standard constructors.
-- need to call constructors explicitly, e.g. :
function torch.classof(obj)
   return torch.factory(torch.typename(obj))
end

-- returns an empty (zero-dim) clone of an obj
function torch.emptyClone(obj)
   return torch.classof(obj)()
end

-- construct an object using a prototype and initialize it with ...
-- won't work with torch.Tensors and other userdata (objects written in C)
-- unless they define __init() constructor.
function torch.protoClone(proto, ...)
   local obj = torch.emptyClone(proto)
   obj:__init(...)
   return obj
end

-- returns a view of a tensor
function torch.view(tensor)
   return torch.emptyClone(tensor):set(tensor)
end

-- simple helpers to serialize/deserialize arbitrary objects/tables
function torch.serialize(object, mode)
   mode = mode or 'binary'
   local f = torch.MemoryFile()
   f = f[mode](f)
   f:writeObject(object)
   local s = f:storage():string()
   f:close()
   return s
end

function torch.deserialize(str, mode)
   mode = mode or 'binary'
   local x = torch.CharStorage():string(str)
   local tx = torch.CharTensor(x)
   local xp = torch.CharStorage(x:size(1)+1)
   local txp = torch.CharTensor(xp)
   txp:narrow(1,1,tx:size(1)):copy(tx)
   txp[tx:size(1)+1] = 0
   local f = torch.MemoryFile(xp)
   f = f[mode](f)
   local object = f:readObject()
   f:close()
   return object
end

-- torch.concat([res], tensors, [dim])
function torch.concat(result, tensors, dim)
   if type(result) == 'table' then
      dim = tensors
      tensors = result
      result = torch.emptyClone(tensors[1])
   end
   dim = dim or 1

   local size
   for i,tensor in ipairs(tensors) do
      if not size then
         size = tensor:size():totable()
      else
         for i,v in ipairs(tensor:size():totable()) do
            if i == dim then
               size[i] = size[i] + v
            else
               assert(size[i] == v, "Cannot concat different sizes")
            end
         end
      end
   end
   
   result:resize(unpack(size))
   local start = 1
   for i, tensor in ipairs(tensors) do
      result:narrow(dim, start, tensor:size(dim)):copy(tensor)
      start = start+tensor:size(dim)
   end
   return result
end

function torch.swapaxes(tensor, new_axes)

   -- new_axes : A table that give new axes of tensor, 
   -- example: to swap axes 2 and 3 in 3D tensor of original axes = {1,2,3}, 
   -- then new_axes={1,3,2}
 
   local sorted_axes = table.copy(new_axes)
   table.sort(sorted_axes)
   
   for k, v in ipairs(sorted_axes) do
      assert(k == v, 'Error: new_axes does not contain all the new axis values')
   end       

   -- tracker is used to track if a dim in new_axes has been swapped
   local tracker = torch.zeros(#new_axes)   
   local new_tensor = tensor

   -- set off a chain swapping of a group of intraconnected dimensions
   _chain_swap = function(idx)
      -- if the new_axes[idx] has not been swapped yet
      if tracker[new_axes[idx]] ~= 1 then
         tracker[idx] = 1
         new_tensor = new_tensor:transpose(idx, new_axes[idx])
         return _chain_swap(new_axes[idx])
      else
         return new_tensor
      end    
   end
   
   for idx = 1, #new_axes do
      if idx ~= new_axes[idx] and tracker[idx] ~= 1 then
         new_tensor = _chain_swap(idx)
      end
   end
   
   return new_tensor
end

function torch.Tensor:dimshuffle(new_axes)
   return torch.swapaxes(self, new_axes)
end
