require 'paths'
require 'torch'

------------------------------------------------------------------------
--[[ dp ]]--
-- dEEp learning library for torch 7, inspired by pylearn2.
------------------------------------------------------------------------


dp = {}
print"dp"
dp.TORCH_DIR = os.getenv('TORCH_DATA_PATH')
dp.DATA_DIR  = paths.concat(dp.TORCH_DIR, 'data')

torch.include('dp', 'datatensor.lua')
torch.include('dp', 'dataset.lua')
torch.include('dp', 'datasource.lua')
torch.include('dp', 'mnist.lua')
torch.include('dp', 'preprocess.lua')
torch.include('dp', 'sampler.lua')
--torch.include('dp', 'propagator.lua')
--torch.include('dp', 'experiment.lua')

