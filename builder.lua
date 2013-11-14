require 'torch'

require 'utils'

------------------------------------------------------------------------
--[[ Builder ]]--
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

--[[ Example design and metadesign
local metadesign = {
   __name='mlp2',
   __metadesign={},
   __require={'dp'},
   experiment={
      __class='dp.Experiment',
      optimizer={__class='dp.StochasticOptimizer'},
      validator={__class='dp.Evaluator'},
      tester={__class='dp.Evaluator'}
   }
}

-- These should be read-only after creation. Or at least, shouldn't mind
-- being called by different models.

local cacheable = {
   __name='Mnist_Standardize',
   __class='dp.Mnist'
}

local design = {
   __metadesign={'mlp2'},
   experiment={
      datasource={
         __cacheable='Mnist_Standardize'
      }      
      model={
      optimizer={
         {
      
         }
      }
      validator={
      
      }
      tester={
      
      }
   }
}
]]--
------------------------------------------------------------------------

local Builder = torch.class("dp.Builder")

function Builder:__init(design_loader)
   -- loads designs into memory (the cache)
   self._design_loader = design_loader
   -- a cache of loaded designs.
   self._design_cache = {}
   -- a cache of cacheable objects (like datasources, datasets)
   -- these are indexed by a unique name
   self._object_cache = {} 
end

-- A recursive function that builds Objects from a design.
-- if the design is a string, it gets the actual design from the loader
function Builder:build(design)
   
end

