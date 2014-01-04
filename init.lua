require 'paths'
require 'torch'
require 'nn'

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
--TODO change this to work without require, with torch.include instead
require "dp/utils"
torch.include('dp', 'multinomial.lua')
torch.include('dp', 'choose.lua')
torch.include('dp', 'xplog.lua')
torch.include('dp', 'mediator.lua')
torch.include('dp', 'objectid.lua')
torch.include('dp', 'eidgenerator.lua')
torch.include('dp', 'analyse.lua')

--[[ data ]]--
torch.include('dp', 'data/datatensor.lua')
torch.include('dp', 'data/dataset.lua')
torch.include('dp', 'data/datasource.lua')
torch.include('dp', 'data/mnist.lua')
torch.include('dp', 'data/preprocess.lua')
torch.include('dp', 'data/batch.lua')
torch.include('dp', 'data/sampler.lua')
torch.include('dp', 'data/cifar10.lua')

--[[ propagator ]]--
torch.include('dp', 'propagator/propagator.lua')
torch.include('dp', 'propagator/optimizer.lua')
torch.include('dp', 'propagator/evaluator.lua')
torch.include('dp', 'propagator/experiment.lua')

--[[ feedback ]]--
torch.include('dp', 'feedback/feedback.lua')
torch.include('dp', 'feedback/compositefeedback.lua')
torch.include('dp', 'feedback/confusion.lua')
torch.include('dp', 'feedback/criteria.lua')

--[[ visitor ]]--
torch.include('dp', 'visitor/visitor.lua')
torch.include('dp', 'visitor/visitorchain.lua')
torch.include('dp', 'visitor/maxnorm.lua')
torch.include('dp', 'visitor/weightdecay.lua')
torch.include('dp', 'visitor/learn.lua')
torch.include('dp', 'visitor/momentum.lua')

--[[ observer ]]--
torch.include('dp', 'observer/observer.lua')
torch.include('dp', 'observer/compositeobserver.lua')
torch.include('dp', 'observer/logger.lua')
torch.include('dp', 'observer/earlystopper.lua')
torch.include('dp', 'observer/savetofile.lua') --not an observer
torch.include('dp', 'observer/learningrateschedule.lua')
torch.include('dp', 'observer/filelogger.lua')

--[[ model ]]--
torch.include('dp', 'model/model.lua')
torch.include('dp', 'model/container.lua')
torch.include('dp', 'model/sequential.lua')
torch.include('dp', 'model/neural.lua')
torch.include('dp', 'model/module.lua')
torch.include('dp', 'model/linear.lua')

--[[ hyper ]]--
torch.include('dp', 'hyper/hyperoptimizer.lua')
torch.include('dp', 'hyper/hyperparamsampler.lua')
torch.include('dp', 'hyper/datasourcefactory.lua')
torch.include('dp', 'hyper/experimentfactory.lua')
torch.include('dp', 'hyper/priorsampler.lua')
torch.include('dp', 'hyper/mnistfactory.lua')
torch.include('dp', 'hyper/mlpfactory.lua')

--[[ postgres ]]--
torch.include('dp', 'postgres/postgres.lua')
torch.include('dp', 'postgres/eidgenerator.lua')
torch.include('dp', 'postgres/logger.lua')
torch.include('dp', 'postgres/xplog.lua')
torch.include('dp', 'postgres/savetofile.lua')
torch.include('dp', 'postgres/earlystopper.lua')
torch.include('dp', 'postgres/done.lua')
torch.include('dp', 'postgres/mlpfactory.lua')
