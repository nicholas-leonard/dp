require 'nnx'
require 'dpnn'
require 'string'
_ = require 'moses'
require 'xlua'
require 'os'
require 'sys'
require 'image'
require 'lfs'
require 'torchx'
ffi = require 'ffi'

------------------------------------------------------------------------
--[[ dp ]]--
-- deep learning library for torch7.
------------------------------------------------------------------------

dp = {}
dp.TORCH_DIR = os.getenv('TORCH_DATA_PATH') or os.getenv('HOME')
   
--[[ utils ]]--
torch.include('dp', 'utils/utils.lua')
torch.include('dp', 'utils/underscore.lua')
torch.include('dp', 'utils/os.lua')
torch.include('dp', 'utils/table.lua')
torch.include('dp', 'utils/torch.lua')

--[[ directory structure ]]--
dp.DATA_DIR = os.getenv('DEEP_DATA_PATH') 
   or paths.concat(dp.TORCH_DIR, 'data')
dp.mkdir(dp.DATA_DIR)

dp.SAVE_DIR = os.getenv('DEEP_SAVE_PATH') 
   or paths.concat(dp.TORCH_DIR, 'save')
dp.mkdir(dp.SAVE_DIR)

dp.LOG_DIR = os.getenv('DEEP_LOG_PATH') 
   or paths.concat(dp.TORCH_DIR, 'log')
dp.mkdir(dp.LOG_DIR)

dp.UNIT_DIR = os.getenv('DEEP_UNIT_PATH') 
   or paths.concat(dp.TORCH_DIR, 'unit')
dp.mkdir(dp.UNIT_DIR)
   
--[[ misc ]]--
torch.include('dp', 'xplog.lua')
torch.include('dp', 'mediator.lua')
torch.include('dp', 'objectid.lua')

--[[ view ]]--
torch.include('dp', 'view/view.lua')
torch.include('dp', 'view/dataview.lua')
torch.include('dp', 'view/imageview.lua')
torch.include('dp', 'view/classview.lua')
torch.include('dp', 'view/sequenceview.lua')
torch.include('dp', 'view/listview.lua')

--[[ dataset ]]--
-- datasets
torch.include('dp', 'data/baseset.lua') -- abstract class
torch.include('dp', 'data/dataset.lua')
torch.include('dp', 'data/sentenceset.lua')
torch.include('dp', 'data/textset.lua')
torch.include('dp', 'data/imageclassset.lua')
torch.include('dp', 'data/batch.lua')

--[[ datasource ]]--
-- generic datasources
torch.include('dp', 'data/datasource.lua')
torch.include('dp', 'data/imagesource.lua')
torch.include('dp', 'data/smallimagesource.lua')
torch.include('dp', 'data/textsource.lua')
-- specific image datasources
torch.include('dp', 'data/mnist.lua')
torch.include('dp', 'data/cifar10.lua')
torch.include('dp', 'data/cifar100.lua')
torch.include('dp', 'data/notmnist.lua')
torch.include('dp', 'data/facialkeypoints.lua')
torch.include('dp', 'data/svhn.lua')
torch.include('dp', 'data/imagenet.lua')
torch.include('dp', 'data/facedetection.lua')
torch.include('dp', 'data/translatedmnist.lua')
-- specific text datasources
torch.include('dp', 'data/billionwords.lua')
torch.include('dp', 'data/penntreebank.lua')

--[[ sampler ]]--
torch.include('dp', 'sampler/sampler.lua')
torch.include('dp', 'sampler/shufflesampler.lua')
torch.include('dp', 'sampler/sentencesampler.lua')
torch.include('dp', 'sampler/randomsampler.lua')
torch.include('dp', 'sampler/textsampler.lua')

--[[ preprocess ]]--
torch.include('dp', 'preprocess/preprocess.lua')
torch.include('dp', 'preprocess/pipeline.lua')
torch.include('dp', 'preprocess/parallelpreprocess.lua')
torch.include('dp', 'preprocess/binarize.lua')
torch.include('dp', 'preprocess/standardize.lua')
torch.include('dp', 'preprocess/gcn.lua')
torch.include('dp', 'preprocess/zca.lua')
torch.include('dp', 'preprocess/lecunlcn.lua')

--[[ propagator ]]--
torch.include('dp', 'propagator/propagator.lua')
torch.include('dp', 'propagator/optimizer.lua')
torch.include('dp', 'propagator/evaluator.lua')
torch.include('dp', 'propagator/experiment.lua')

--[[ feedback ]]--
torch.include('dp', 'feedback/feedback.lua')
torch.include('dp', 'feedback/compositefeedback.lua')
torch.include('dp', 'feedback/confusion.lua')
torch.include('dp', 'feedback/perplexity.lua')
torch.include('dp', 'feedback/topcrop.lua')
torch.include('dp', 'feedback/fkdkaggle.lua')
torch.include('dp', 'feedback/facialkeypointfeedback.lua')

--[[ observer ]]--
torch.include('dp', 'observer/observer.lua')
torch.include('dp', 'observer/compositeobserver.lua')
torch.include('dp', 'observer/logger.lua')
torch.include('dp', 'observer/errorminima.lua')
torch.include('dp', 'observer/earlystopper.lua')
torch.include('dp', 'observer/savetofile.lua') --not an observer (but used in one)
torch.include('dp', 'observer/adaptivedecay.lua')
torch.include('dp', 'observer/filelogger.lua')
torch.include('dp', 'observer/hyperlog.lua')

--[[ nn ]]--
torch.include('dp', 'nn/Print.lua')
torch.include('dp', 'nn/FairLookupTable.lua')

--[[ test ]]--
torch.include('dp', 'test/test.lua')
torch.include('dp', 'test/test-cuda.lua')
torch.include('dp', 'test/test-datasets.lua')

return dp
