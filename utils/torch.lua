--useful for validating if an object is an instance of a class, 
--even when the class is a super class.
--e.g pattern = "^torch[.]%a*Tensor$"
--typepattern(torch.Tensor(4), pattern) and
--typepattern(torch.DoubleTensor(5), pattern) are both true.
--typepattern(3, pattern)
function torch.typepattern(obj, pattern)
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
-- DEPRECATED:
typepattern = torch.typepattern
-- END

function torch.typeString_to_tensorType(type_string)
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
   local obj_t = torch.typename(obj)
   if not obj_t then
      error("type"..torch.type(obj).."has no torch.class registered constructor", 2)
   end
   local modula = string.tomodule(obj_t)
   return modula
end

-- returns an empty clone of an obj
function torch.emptyClone(obj)
   error("Deprecated use torch.protoClone instead", 2)
end

-- construct an object using a prototype and initialize it with ...
-- works with any class registered with torch.class
function torch.protoClone(proto, ...)
   local class = torch.classof(proto)
   return class(...)
end

-- torch.concat([res], tensors, [dim])
function torch.concat(result, tensors, dim, index)
   index = index or 1
   if type(result) == 'table' then
      index = dim or 1
      dim = tensors
      tensors = result
      result = torch.protoClone(tensors[index])
   elseif result == nil then
      assert(type(tensors) == 'table', "expecting table at arg 2")
      result = torch.protoClone(tensors[index])
   end
   
   assert(_.all(tensors, 
      function(k,v) 
         return torch.isTensor(v) 
      end),
      "Expecting table of torch.tensors at arg 1 and 2 : "..torch.type(result)
   )
   
   dim = dim or 1

   local size
   for i,tensor in ipairs(tensors) do
      if not size then
         size = tensor:size():totable()
         size[dim] = 0
      end
      for j,v in ipairs(tensor:size():totable()) do
         if j == dim then
            size[j] = (size[j] or 0) + v
         else
            if size[j] and size[j] ~= v then
               error(
                  "Cannot concat dim "..j.." with different sizes: "..
                  (size[j] or 'nil').." ~= "..(v or 'nil')..
                  " for tensor at index "..i, 2
               )
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


----------- FFI stuff used to pass storages between threads ------------
-- https://raw.githubusercontent.com/facebook/fbcunn/master/examples/imagenet/util.lua

ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THIntStorage_free(THIntStorage *self);
]]

function torch.setFloatStorage(tensor, storage_p, free)
   assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
   if free then
      local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
      if cstorage ~= nil then
         ffi.C['THFloatStorage_free'](cstorage)
      end
   end
   local storage = ffi.cast('THFloatStorage*', storage_p)
   tensor:cdata().storage = storage
end

function torch.setIntStorage(tensor, storage_p, free)
   assert(storage_p and storage_p ~= 0, "IntStorage is NULL pointer");
   if free then 
      local cstorage = ffi.cast('THIntStorage*', torch.pointer(tensor:storage()))
      if cstorage ~= nil then
         ffi.C['THIntStorage_free'](cstorage)
      end
   end
   local storage = ffi.cast('THIntStorage*', storage_p)
   tensor:cdata().storage = storage
end

function torch.sendTensor(tensor)
   local size = tensor:size()
   local ttype = tensor:type()
   local storage =  tonumber(ffi.cast('intptr_t', torch.pointer(tensor:storage())))
   tensor:cdata().storage = nil
   return {storage, size, ttype}
end

function torch.receiveTensor(tensor, pointer, size, ttype, free)
   if tensor then
      tensor:resize(size)
      assert(tensor:type() == ttype, 'Tensor is wrong type')
   else
      tensor = torch[ttype].new():resize(size)      
   end
   if ttype == 'torch.FloatTensor' then
      torch.setFloatStorage(tensor, pointer, free)
   elseif ttype == 'torch.IntTensor' then
      torch.setIntStorage(tensor, pointer, free)
   else
      error('Unknown type')
   end
   return buffer
end

-- creates or loads a checkpoint at path.
-- factory is a function that returns an object. 
-- If a file exists at location path, then it is loaded and returned.
-- Otherwise, the factory is executed and its result is saved 
-- at location path for later.
-- This function is usuful for optimizing slow DataSource loadtimes by
-- caching their results to disk for later.
function torch.checkpoint(path, factory, overwrite, mode)
   if paths.filep(path) and not overwrite then
      return torch.load(path, mode)
   end
   local obj = factory()
   paths.mkdir(paths.dirname(path))
   torch.save(path, obj, mode)
   return obj
end
