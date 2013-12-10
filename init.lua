require 'paths'
require 'torch'

------------------------------------------------------------------------
--[[ dp ]]--
-- deep learning library for torch 7, inspired by pylearn2.
------------------------------------------------------------------------


dp = {}

dp.TORCH_DIR = os.getenv('TORCH_DATA_PATH')

dp.DATA_DIR = os.getenv('DEEP_DATA_PATH') 
   or paths.concat(dp.TORCH_DIR, 'data')

dp.SAVE_DIR = os.getenv('DEEP_SAVE_PATH') 
   or paths.concat(dp.TORCH_DIR, 'save')
   
dp.LOG_DIR = os.getenv('DEEP_LOG_PATH') 
   or paths.concat(dp.TORCH_DIR, 'log')

--[[ misc ]]--
require "dp/utils"
torch.include('dp', 'multinomial.lua')
torch.include('dp', 'choose.lua')
torch.include('dp', 'mediator.lua')
torch.include('dp', 'query.lua')

--[[ data ]]--
torch.include('dp', 'data/datatensor.lua')
torch.include('dp', 'data/dataset.lua')
torch.include('dp', 'data/datasource.lua')
torch.include('dp', 'data/mnist.lua')
torch.include('dp', 'data/preprocess.lua')
torch.include('dp', 'data/batch.lua')
torch.include('dp', 'data/sampler.lua')


--[[ propagator ]]--
torch.include('dp', 'propagator/propagator.lua')
torch.include('dp', 'propagator/optimizer.lua')
torch.include('dp', 'propagator/evaluator.lua')
torch.include('dp', 'propagator/experiment.lua')

--[[ feedback ]]--
torch.include('dp', 'feedback/feedback.lua')
torch.include('dp', 'feedback/confusion.lua')

--[[ visitor ]]--
torch.include('dp', 'visitor/visitor.lua')

--[[ observer ]]--
torch.include('dp', 'observer/observer.lua')
torch.include('dp', 'observer/logger.lua')


--[[ model ]]--
torch.include('dp', 'model/model.lua')
torch.include('dp', 'model/module.lua')
torch.include('dp', 'model/sequential.lua')

--[[ hyper ]]--
torch.include('dp', 'hyper/hyperoptimizer.lua')

--[[ postgres ]]--
torch.include('dp', 'postgres/postgres.lua')
torch.include('dp', 'postgres/eidgenerator.lua')
torch.include('dp', 'postgres/logger.lua')
torch.include('dp', 'postgres/query.lua')
torch.include('dp', 'postgres/savetofile.lua')
torch.include('dp', 'postgres/earlystopper.lua')
torch.include('dp', 'postgres/done.lua')
