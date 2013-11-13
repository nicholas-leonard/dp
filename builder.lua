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
   name='mlp2',
   metadesign={},
   require={'dp'},
   experiment={'dp.Experiment',
      optimizer={'dp.StochasticOptimizer'},
      validator={'dp.Evaluator'},
      tester={'dp.Evaluator'},
   }
}

local design = {
   metadesign={'mlp2'},
   experiment={
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

function Builder:__init(loader)
   -- loads designs into memory (the cache)
   self._loader = loader
   -- contains the loaded designs.
   self._cache
end

-- A recursive function that builds Objects from a design.
-- if the design is a string, it gets the actual design from the loader
function Builder:build(design)
   
end

