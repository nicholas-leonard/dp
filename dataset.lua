require 'torch'
require 'xlua'


local Dataset = torch.class("data.Dataset")

--[[
TODO
Iterators (shuffled, etc)
Refactor preprocessing
Eliminate non-standard dependencies
]]--


-- Wraps a table containing a dataset to make it easy to transform the dataset
-- and then sample from.  Each property in the data table must have a tensor or
-- table value, and then each sample will be retrieved by indexing these values
-- and returning a single instance from each one.
--
-- e.g.
--
--   -- a 'dataset' of random samples with random class labels
--   data_table = {
--     data  = torch.Tensor(10, 20, 20),
--     classes = torch.randperm(10)
--   }
--   metadata = { name = 'random', classes = {1,2,3,4,5,6,7,8,9,10} }
--   dataset = Dataset(data_table, metadata)
--
function Dataset:__init()
   local args
   args, self.inputs, self.targets, self.topological, self.axes, 
         self.view_converter
      = xlua.unpack(
      {...},
      'Dataset constructor', nil,
      {arg='inputs', type='table', 
       help=[[Inputs of the dataset taking the form of torch.Tensor with
            2 dimensions, or more if topological is true. Alternatively,
            inputs may take the form of a table of such torch.Tensors.
            The first dimension of the torch.Tensor(s) should be of size
            number of examples.
            ]], req=true},
      {arg='targets', type='table', 
       help=[[Targets of the dataset taking the form of torch.Tensor
            with 1-2 dimensions. Alternatively, targets may take the 
            form of a table of such torch.Tensors. The first dimension 
            of the torch.Tensor(s) should be of size number of examples.
            ]], default=nil},
      {arg='topological', type='boolean', 
       help=[[This should be true if the provided inputs are topological
            ]], default=nil},
      {arg='axes', type='table', 
       help=[[A table defining the order and nature of each dimension
            of a batch of images. An example would be {'b', 0, 1,'c'}, 
            where the dimensions represent a batch of examples :'b', 
            the first horizontal axis of the image : 0, the vertical 
            axis : 1, and the color channels : 'c'.
            ]], default={'b','x','y','c'}},
   )
   
   inputs, targets, axes, topo_view, 
   self.inputs = inputs
   self.targets = targets

   global_metadata = global_metadata or {}

   self._name = global_metadata.name
   self._classes = global_metadata.classes or {}
end

function Dataset:whichSet()
   return self.which_set
end

function Dataset:isTrain()
   if self.which_set == 'train' then
      return true
   end
   return false
end

function Dataset:setInputs(inputs)
   self.inputs = inputs
end

function Dataset:setTargets(targets)
   self.targets = targets
end


-- Returns the number of samples in the dataset.
function Dataset:size()
   return self.dataset.data:size(1)
end

function Dataset:inputs()
   return self.dataset.data
end

function Dataset:targets()
   return self.dataset.class
end


-- Returns the dimensions of a single sample as a table.
-- e.g.
--   mnist          => {1, 28, 28}
--   natural images => {3, 64, 64}
function Dataset:dimensions()
   local dims = self.dataset.data:size():totable()
   table.remove(dims, 1)
   return dims
end


-- Returns the total number of dimensions of a sample.
-- e.g.
--   mnist => 1*28*28 => 784
function Dataset:n_dimensions()
   return fn.reduce(fn.mul, 1, self:dimensions())
end


-- Returns the classes represented in this dataset (if available).
function Dataset:classes()
   return self._classes
end


-- Returns the string name of this dataset.
function Dataset:name()
   return self._name
end


-- Returns the specified sample (a table) by index.
--
--   sample = dataset:sample(100)
function Dataset:sample(i)
    local sample = {}

    for key, v in pairs(self.dataset) do
        sample[key] = v[i]
    end

    return sample
end



-- Returns an infinite sequence of data samples.  By default they
-- are shuffled samples, but you can turn shuffling off.
--
--   for sample in seq.take(1000, dataset:sampler()) do
--     net:forward(sample.data)
--   end
--
--   -- turn off shuffling
--   sampler = dataset:sampler({shuffled = false})
--
--   -- generate animations over 10 frames for each sample, which will
--   -- randomly rotate, translate, and/or zoom within the ranges passed.
--   local anim_options = {
--      frames      = 10,
--      rotation    = {-20, 20},
--      translation = {-5, 5, -5, 5},
--      zoom        = {0.6, 1.4}
--   }
--   s = dataset:sampler({animate = anim_options})
--
--   -- pass a custom pipeline for post-processing samples
--   s = dataset:sampler({pipeline = my_pipeline})
--
function Dataset:sampler(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices
   local size = self:size()

   local pipeline, pipe_size = pipe.construct_pipeline(options)

   local function make_sampler()
       if shuffled then
           indices = torch.randperm(size)
       else
           indices = seq.range(size)
       end

       local sample_seq = seq.map(fn.partial(self.sample, self), indices)

       if options.animate then
          sample_seq = animate(options.animate, sample_seq)
       end

       if pipe_size > 0 then
          sample_seq = seq.map(pipeline, sample_seq)
       end

       if options.pipeline then
          sample_seq = seq.map(options.pipeline, sample_seq)
       end

       return sample_seq
    end

   return seq.flatten(seq.cycle(seq.repeatedly(make_sampler)))
end


