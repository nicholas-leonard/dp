require 'torch'

------------------------------------------------------------------------
--[[ HyperOptimizer ]]--
-- Constructs dp/torch objects from a design table
-- A design table can specify metadesign tables that will be used 
-- as metatables (which provide default values) for design 
-- or other metadesign tables.
-- metadesign have the same syntax as design tables, such that they
-- can be stored in the same SQL table.
-- Each design has a unique name which is used to index 
-- the table. designs adopt the name of the metadesign + a random key
-- used to maintain uniqueness of each design. This name is 
-- also used to initialize the experiment. Which can then be used 
-- to match the design with any generated object, like results.
-- Using the design table and its metadesign tables, we have a 
-- structured view of all experiments, such that these can be passed 
-- to analysis query functions, along with results, to compare different
-- experiments.

-- A design is simply a table having some reserved keys :
-- name, metadesign and require. All others specify objects that 
-- will be returned to the caller of build().
-- The returned objects will be stored in a table with the same
-- keys. So in the bellow example, {experiment=experiment_instance}
-- would be returned. 

-- Not serializable
------------------------------------------------------------------------
local HyperOptimizer = torch.class("HypeOptimizer")
HyperOptimizer.isHyperOptimizer = true

function HyperOptimizer:__init(...)
   local args, group, loader = xlua.unpack(
      {... or {}},
      'HyperOptimizer', nil,
      {arg='group', type='string', req=true,
       help='identifies a group of experiments'},
      {arg='loader', type='dp.Loader' req=true,
       help='loads designs and metadesigns from a file, database, etc.'}
   )
   self._group = group
   -- loads designs into memory (the cache)
   self._loader = loader
   -- a cache of cacheable objects (like datasources, datasets)
   -- these are indexed by a unique name
   self._object_cache = {} 
end

function HyperOptimizer:run()
   while true
      --sample hyper-parameters (one of them is design?)
   end
end


function HyperOptimizer:setmetadesigns(design)
   
end

local keywords = {
   '__name', '__require', '__metadesign', '__cache', '__sample',
   '__hyper', '__class'
}

local innerwords = {
   '__cache', '__sample', '__class'
}
   

-- A recursive function that builds Objects from a design.
-- if the design is a string, it gets the actual design from the loader
function HyperOptimizer:build(design, isMeta)
   
   -- require requirements
   if requirements then
      requires = _.concat(requires, requirements)
      for i, k in ipairs(requirements) do
         require(k)
      end
   end
   
   -- if design is metadesign, don't build, just return
   if isMeta then
      return design
   end
   build = {}
   for name, construct in pairs(design) do
      if not _.contains(self._keywords, name) then
         build[name] = self:build(construct, false, metadesigns)
         
      end
   end
   return build   
end

function HyperOptimizer:loadMetaDesign(metadesign_name)
   self._pg
end


------------------------------------------------------------------------
--[[ DesignFactory ]]--
------------------------------------------------------------------------
local DesignFactory = torch.class("dp.DesignFactory")

function DesignFactory:__init(...)
   local args, loader, hyperopt = xlua.unpack(
      {... or {}}, 
      'DesignFactory', nil,
      {arg='loader', type='dp.Loader', req=true},
      {arg='hyperopt', type='dp.HyperOptimizer'}
   )
   self._loader = loader
   self._hyperopt = hyperopt
end

function DesignFactory:create(design)
   local root = dp.RootDesign(design)
   root:setup(self._loader)
   -- create a plan from a design
   -- initialize a tree of designs and values recursively
   local function factory(design, samples)
      local plan = {}
      for k, v in pairs(design:design()) do
         if (type(v) == "table") then
            if (v.__sample) then
               local obj = dp.SamplerDesign(v)
               obj:setup(self._loader, factory, self._hyperopt)
               plan[k] = obj
            elseif (v.__class) then
               local obj = dp.ClassDesign(v)
               obj:setup(self._loader)
               plan[k] = obj
            elseif not torch.typename(v) then
               plan[k] = factory(v)
            else
               plan[k] = v
            end
         else
            plan[k] = v
         end
      end
      return plan
   end
   plan = factory(root:design())
   design:setPlan(plan)
   return design
end


------------------------------------------------------------------------
--[[ Design ]]--
------------------------------------------------------------------------
local Design = torch.class("dp.Design")
Design.isDesign = true

function Design:__init(design)
   if type(design) == 'string' then
      design = torch.deserialize(design)
   end
   local args, name = xlua.unpack(
      {design},
      'Design', nil,
      {arg='__name', type='string | number'}
   )
   self._name = name
   design.__name = nil
   self._design = design
end

function Design:name()
   return self._name
end

function Design:design()
   return self._design
end

function Design:setup(loader, factory, hyperopt)
   self._plan = factory(self._design)
   self._setup = true
end

--[[function Design:setPlan(plan)
   self._plan = plan
end

function Design:plan()
   return self._plan
end]]--

function Design:build(...)
   assert(self._setup)
   local obj = {}
   local inner_obj
   -- each design is responsible for building its sub-designs
   for k,v in pairs(self._plan) do
      if type(v) == 'table' then
         if v.isDesign then
            obj[k] = v:build(...)
         end
      else
         obj[k] = v
      end
   end
   return obj
end

function Design:tostring()
   print(self._design)
end

local function getGlobal(class_str)
   local class = _G
   for i, v in ipairs(_.split(class_str, '[.]')) do
      class = class[v]
   end
   return class
end   

------------------------------------------------------------------------
--[[ RootDesign ]]--
------------------------------------------------------------------------
local RootDesign = torch.class("dp.RootDesign")
RootDesign.isRootDesign= true

function RootDesign:__init(design)
   Design.__init(self, design)
   local args, md_names, requires = xlua.unpack(
      {self._design},
      'Design', nil,
      {arg='__metadesign', type='table'},
      {arg='__require', type='string'}
   )
   self._md_names = md_names
   self._requires = requires
   -- delete special key-value pairs :
   self._design.__metadesign = nil
   self._design.__require = nil
   self._metadesigns = {}
end

function RootDesign:mdNames()
   return self._md_names
end

function RootDesign:requires()
   return self._requires
end

-- require requirements
function RootDesign:require()
   if self._required then return end
   for i, r in ipairs(self._requires) do
      require(r)
   end
   for i, md in ipairs(self._metadesigns) do
      md:require()
   end
   self._required = true
end

function RootDesign:setup(loader, factory, hyperopt)
   -- load metadesigns using loader
   if self._md_names then
      for i, md_name in ipairs(self._md_names) do
         local md = loader:loadMetaDesign(md_name)
         md:setup(loader)
         table.insert(self._metadesigns, md)
      end
   end
   -- coalesce metadesigns into design
   for i, metadesign in ipairs(self._metadesigns) do
      table.merge(self._design, metadesign:design())
   end
   -- build sub-designs
   Design.setup(loader, factory, hyperopt)
end

function RootDesign:build(...)
   self:require()
   return Design.build(self, ...)
end

------------------------------------------------------------------------
--[[ SamplerDesign ]]--
------------------------------------------------------------------------
local SamplerDesign = torch.class("dp.SamplerDesign")
SamplerDesign.isSamplerDesign = true

function SamplerDesign:__init(design)
   Design.__init(self, design)
   local args, sampler = xlua.unpack(
      {design},
      'SamplerDesign', nil,
      {arg='__sample', type='string'}
   )
   -- class name of sampler
   self._sampler = sampler
   -- delete special key-value pairs :
   design.__sampler = nil
   self._design = {}
   for k, v in pairs(design) do
      
   end
end

function SamplerDesign:setup(loader, factory, hyperopt)
   
end

-- Samples a concrete design
function SamplerDesign:build(hyper, loader)
   assert(self._setup, "Design Error: design must be setup. " ..
         "\nHint: Provide a loader to Design:sample(hyper, loader).")
   local function sample(design, sample)
      sample = sample or {}
      local samplers = {}
      for k, v in pairs(design) do
         local sample_v = v
         if (type(v) == "table") then
            if (v.isParamSampler) then
               sample_v = v:sample()
            elseif not torch.typename(v) then
               sample_v = init_samplers(v)
            end
         end
         sample[k] = sample_v
      end
      return sample
   end
   return sample(design)
end

------------------------------------------------------------------------
--[[ PGLoader ]]--
-- Used to load objects stored in a postgresql database
------------------------------------------------------------------------
local PGLoader = torch.class("PGLoader")

function PGLoader:__init()
   -- a cache of loaded designs.
   self._design_cache = {}
   self._pg = dp.Postgres()
end

function PGLoader:loadDesign(design_id, dont_cache)
   local design = self._design_cache[design_id]
   if design then
      return design
   end
   local row = self._pg:fetchOne(
      "SELECT design_Choosele " ..
      "FROM dp.design " ..
      "WHERE design_id = %s",
      {design_id},
      "a"
   )
   design = dp.Design(row.design_Choosele)
   -- cache it for later use
   if not dont_cache then
      self._design_cache[design_id] = design
   end
   return design, requires
end

function PGLoader:loadMetaDesign(md_name, dont_cache)
   local md = self._design_cache[md_name]
   if md then
      return md
   end
   local row = self._pg:fetchOne(
      "SELECT md_Choosele " ..
      "FROM dp.metadesign " ..
      "WHERE md_name = %s",
      {md_name},
      "a"
   )
   md = dp.Design(row.md_Choosele)
   -- cache it for later use
   if not dont_cache then
      self._design_cache[md_name] = md
   end
   return md
end

------------------------------------------------------------------------
-- TEST --
--[[
The idea here is to build a grid search as a tree, where 
each variable-value has a prior probability of being sampled.
Each sampling hyper-param is thus unique across different architectures.
All distributions have discrete values, which allows for the 
accumulation of error statistics for each value. 
All sample probability distributions provide a prior to the hyper-opt
process.
Each hyper-parameter has a name.

Metadesigns allow for reusing of components accross designs.
Metadesigns are just modules. 
Metadesigns have little use other than saving on storage space?
For each concrete experiment, we need to store its design and reports.
There will be very few designs, at least much less than reports, since 
each experiment has only one. So we shouldn't pre-mature optimize 
design storage with metadesigns. However, we can think of the 
distribution of designs as a kind of metadesign. For the purpose of 
reporting, these design distributions (meta designs) should be stored
in there own table for later use. They should also have a name.

Learning curves should be able to identify each experiment with the 
values of its sampled hyper-parameters. SamplerDesigns can have 
an optional name passed to them or they can be named with their position
in the tree.

What if we want to sample different parameters globally, so for example
sample the activation function to be used in different layers once for 
all layers? We would need to define the sample in a higher level, name 
it, and reference it by name. Function factory(design) would be 
modified to factory(design, samples) to allow referencing designs to 
be initialized with the sample. We could call it a SharedSamplerDesign
or something.

We code a design and pass it to the hyper-optimizer which will use it 
to sample experiments. A design is just a way to specify a hierarchy 
of constructors. Its also a way to specify a hierarchical distribution
of hyper parameters from which to sampler from to perform a kind of 
random-grid search.

Designs, hyper-params and results will be stored in a database.
For now, the hyper-opt process will take the form of the experimentation
loop where we hypothesis, experiment, observe and so on.

Hyperoptimizer needs to explore and exploit the design distribution 
in order to optimize it. The reward is the accuracy of the experiment.
Hyperopt would sample a design from the design distribution. The 
design would be stored until the experiment finished and the design 
reward is known. Each random variable and possible value in the tree 
would be associated to a binary visible unit of a generative model, 
which would be trained and sampled from hierarchically, as in the tree.
In this particular model, the hyper-params would be sampled from a 
model and object external to the design. Or SamplerDesigns would be 
initialized with this model, and this model would be setup with them. 
Or the RootDesign would have this model and build it based on 
GenerativeSamplerDesign which would have nothing but a name and possible
values. Its basically the third term we wish to pass to Design:setup(), 
which designs could use themselves setup and reference to sample their 
design. The first stage of the sampling process could be to sample from 
the generative model, and then build the design by traversing the tree
top down to generate a concrete design, where GenerativeDesigns would 
simply call the generative model with the name of their sample.

In the interest of providing for such future work, we could make all 
samplers call a shared object with their name as argument to get a 
sample. And all hyper-params could be defined in this model. It might 
make the code more readable. But again, what about when the possible 
values are designs? All random values are defined in the design tree,
and the shared-object simply provides an index.

What about when we have a dependency within a layer of the tree? 
For example, the preprocessing should be different for a convolution 
vs a densely connected layer? Or a model using ReLU should have a 
smaller learning rate? Etc. We would require a kind of bayesian network
to model the dependencies, where the trueness of one event implies 
a probability of trueness of other events.

Design could just be a template with keywords for filling in 
parameters. But then this wouldn't allow for dependencies. Well we could
implement a MultiDesign which would choose a design based on a sampled
hyper-parameter configuration. But then the hyper-parameter sampler 
would have to model and match these dependencies.

Requirements :
 - SharedSamplerDesign
 - Store design distributions in database
 - Print a design for reporting.
 - Store design (hyper-parameter configuration) for later reporting
   and analysis. 
 - No table version, just objects
 - Cacheable designs

Bonus :
 - HyperOptimizer can update probabilities of samplers
--]]
------------------------------------------------------------------------
local function test()
   -- Example of hyper-param config
   --[[local hg = HyperGraph()
   hg:add(
      HyperNode{
         name = 'learning_rate', 
         dist = dp.Choose{0.1, 0.01, 0.001, 0.0001}
      }
   )
   hg:add(
      HyperNode{ 
         name = 'preprocess',
         dist = dp.WeigthedChoose{
            values={'std', 'etc'}, 
            weights={0.2, 0.8}
         }
      }
   )
   hg:add(
      HyperNode{ 
         name = 'model_type',
         dist = dp.WeigthedChoose{
            values={'conv', 'cudaconv', '}, 
            weights={0.2, 0.8}
         }
      }
   )
   hg:add(
      HyperArrow{
         tail = {'preprocess','std'},
         dist
      }
   )--]]
   -- Example design and metadesign
   local metadesign = {
      __name='mlp2',
      __metadesign={},
      __require={'dp'},
      experiment={
         __class='dp.Experiment',
         optimizer={__class='dp.Optimizer'},
         validator={__class='dp.Evaluator'},
         tester={__class='dp.Evaluator'}
      }
   }

   -- These should be read-only after creation. Or at least, shouldn't mind
   -- being called by different models.

   local cache = {
      __name='Mnist_Standardize',
      __class='dp.Mnist'
   }

   local design = {
      __metadesign={'mlp2'},
      datasource={
         __cache='Mnist_Standardize',
         __class='dp.Mnist',
         input_preprocess={
               __class='dp.Standardize'
            }
      }
      experiment={
         learning_rate = {_
            __sample = 'dp.Choose',
            options = {0.1, 0.01, 0.001, 0.0001}
         }
         --learning_rate = dp.sampleMultinomialFrom{[0.1] = 0.05,  0.01, 0.001, 0.0001} 
         model={ __sample = 'dp.WeightedChoose'
               options = {
                  { __class='dp.Linear'
                    input_size = {
                        
                     }
                  }
               },
               weights = {
                  
               }
         }
         optimizer={
            {
               sampler={
                  __class='dp.ShuffleSampler',
                  batch_size = {
                     __sample = 'dp.weightedChoose',
                     [32] = 0.5,
                     [8] = 0.1,
                     [64] = 0.2
                  }
               }
            }
         }
         validator={
         
         }
         tester={
         
         }
      }
   }
   
   local design = dp.RootDesign{
      datasource = dp.CachedDesign{
         name ='Mnist_Standardize',
      }
      experiment = dp.ClassDesign{__class='dp.Experiment',
         learning_rate = dp.SampledDesign{__class='dp.Choose',
            values = {0.1, 0.01, 0.001, 0.0001}
         }
         __named = {
            hidden1_size = dp.SampleDesign{__class='dp.WeightedChoose',
               dist = {[32] = 0.1, [64] = 0.1, [128] = 0.1,
                       [256] = 0.3, [512] = 0.5, [1024] = 0.1}
         }
         --learning_rate = dp.sampleMultinomialFrom{[0.1] = 0.05,  0.01, 0.001, 0.0001} 
         model = dp.ClassDesign{__class'dp.Sequential',
            
               hidden2_size = dp.SampleDesign{__class='dp.WeightedChoose',
                     dist = {[
            }
            {
               dp.ClassDesign{__class='dp.Linear',
                  input_size=datasource._feature_size, 
                  output_size=dp.NamedDesign{
               },
               dp.ClassDesign{__class=
            }
         }
         optimizer={
            {
               sampler={
                  __class='dp.ShuffleSampler',
                  batch_size = {
                     __sample = 'dp.weightedChoose',
                     [32] = 0.5,
                     [8] = 0.1,
                     [64] = 0.2
                  }
               }
            }
         }
         validator={
         
         }
         tester={
         
         }
      }
   }
   
   --]]
end


------------------------------------------------------------------------
--[[ ParamSampler ]] --
-- Interface
------------------------------------------------------------------------ 
local ParamSampler = torch.class("dp.ParamSampler")
ParamSampler.isParamSampler = true

function ParamSampler:sample()
   error"Sample Error : method sample() not implemented"
end

------------------------------------------------------------------------
--[[ Choose ]] --
-- Uniformly Chooses one of the provided options.
------------------------------------------------------------------------  
local Choose = torch.class("dp.Choose", "dp.ParamSampler")

function Choose:__init(options)
   self._options
end

function Choose:sample()
   return self._options[math.random(#self._options)]
end

------------------------------------------------------------------------
--[[ WeightedChoose ]] --
-- Choose options by sampling from a multinomial probability distribution
-- proportional to their respective weight.
------------------------------------------------------------------------  
local WeightedChoose = torch.class("dp.WeightedChoose", "dp.ParamSampler")

-- Distribution is a table of where keys are options, 
-- and their values are weights.
function WeightedChoose:__init(distribution)
   self._size = _.size(distribution)
   self._probs = torch.DoubleTensor(self._size)
   self._options = {}
   local i = 0
   for k,v in pairs(distribution) do
      self._probs[i] = v
      self._options[i] = k
   end
end

function WeightedChoose:sample()
   local index = dp.multinomial(self._probs)[1]
   return self._options[index]
end

