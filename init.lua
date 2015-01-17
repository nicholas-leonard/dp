require 'torch'
require 'nn'
require 'nnx'
require 'string'
_ = require 'moses'
require 'xlua'
require 'fs'
require 'os'
require 'sys'
require 'image'
require 'lfs'

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
dp.check_and_mkdir(dp.DATA_DIR)

dp.SAVE_DIR = os.getenv('DEEP_SAVE_PATH') 
   or paths.concat(dp.TORCH_DIR, 'save')
dp.check_and_mkdir(dp.SAVE_DIR)

dp.LOG_DIR = os.getenv('DEEP_LOG_PATH') 
   or paths.concat(dp.TORCH_DIR, 'log')
dp.check_and_mkdir(dp.LOG_DIR)

dp.UNIT_DIR = os.getenv('DEEP_UNIT_PATH') 
   or paths.concat(dp.TORCH_DIR, 'unit')
dp.check_and_mkdir(dp.UNIT_DIR)
   
--[[ misc ]]--
torch.include('dp', 'choose.lua')
torch.include('dp', 'xplog.lua')
torch.include('dp', 'mediator.lua')
torch.include('dp', 'objectid.lua')
torch.include('dp', 'node.lua')

--[[ view ]]--
torch.include('dp', 'view/view.lua')
torch.include('dp', 'view/dataview.lua')
torch.include('dp', 'view/imageview.lua')
torch.include('dp', 'view/classview.lua')
torch.include('dp', 'view/sequenceview.lua')
torch.include('dp', 'view/listview.lua')

--[[ data ]]--
torch.include('dp', 'data/baseset.lua')
torch.include('dp', 'data/dataset.lua')
torch.include('dp', 'data/sentenceset.lua')
torch.include('dp', 'data/imageclassset.lua')
torch.include('dp', 'data/batch.lua')
torch.include('dp', 'data/carry.lua')

torch.include('dp', 'data/datasource.lua')
torch.include('dp', 'data/mnist.lua')
torch.include('dp', 'data/cifar10.lua')
torch.include('dp', 'data/cifar100.lua')
torch.include('dp', 'data/notmnist.lua')
torch.include('dp', 'data/facialkeypoints.lua')
torch.include('dp', 'data/billionwords.lua')
torch.include('dp', 'data/svhn.lua')
torch.include('dp', 'data/imagenet.lua')

--[[ sampler ]]--
torch.include('dp', 'data/sampler.lua')
torch.include('dp', 'data/shufflesampler.lua')
torch.include('dp', 'data/sentencesampler.lua')
torch.include('dp', 'data/randomsampler.lua')

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
torch.include('dp', 'feedback/fkdkaggle.lua')
torch.include('dp', 'feedback/facialkeypointfeedback.lua')
--torch.include('dp', 'feedback/criteria.lua')

--[[ visitor ]]--
torch.include('dp', 'visitor/visitor.lua')
torch.include('dp', 'visitor/visitorchain.lua')
torch.include('dp', 'visitor/recurrentvisitorchain.lua')
torch.include('dp', 'visitor/maxnorm.lua')
torch.include('dp', 'visitor/weightdecay.lua')
torch.include('dp', 'visitor/learn.lua')
torch.include('dp', 'visitor/momentum.lua')

--[[ observer ]]--
torch.include('dp', 'observer/observer.lua')
torch.include('dp', 'observer/compositeobserver.lua')
torch.include('dp', 'observer/logger.lua')
torch.include('dp', 'observer/errorminima.lua')
torch.include('dp', 'observer/earlystopper.lua')
torch.include('dp', 'observer/savetofile.lua') --not an observer
torch.include('dp', 'observer/learningrateschedule.lua')
torch.include('dp', 'observer/adaptivelearningrate.lua')
torch.include('dp', 'observer/filelogger.lua')

--[[ nn ]]--
torch.include('dp', 'nn/Print.lua')
torch.include('dp', 'nn/PrintSize.lua')
torch.include('dp', 'nn/FairLookupTable.lua')

--[[ model ]]--
torch.include('dp', 'model/model.lua')
torch.include('dp', 'model/container.lua')
torch.include('dp', 'model/sequential.lua')
torch.include('dp', 'model/layer.lua')
torch.include('dp', 'model/neural.lua')
torch.include('dp', 'model/module.lua')
torch.include('dp', 'model/dictionary.lua')
torch.include('dp', 'model/narrowdictionary.lua')
torch.include('dp', 'model/recurrentdictionary.lua')
torch.include('dp', 'model/softmaxtree.lua')
torch.include('dp', 'model/softmaxforest.lua')
torch.include('dp', 'model/mixtureofexperts.lua')
torch.include('dp', 'model/blocksparse.lua')
torch.include('dp', 'model/convolution1D.lua')
torch.include('dp', 'model/convolution2D.lua')
torch.include('dp', 'model/inception.lua')

--[[ loss ]]--
torch.include('dp', 'loss/loss.lua')
torch.include('dp', 'loss/nll.lua')
torch.include('dp', 'loss/kldivergence.lua')
torch.include('dp', 'loss/treenll.lua')
torch.include('dp', 'loss/null.lua')
torch.include('dp', 'loss/criterion.lua')

--[[ hyper ]]--
torch.include('dp', 'hyper/hyperoptimizer.lua')
torch.include('dp', 'hyper/hyperparamsampler.lua')
torch.include('dp', 'hyper/datasourcefactory.lua')
torch.include('dp', 'hyper/experimentfactory.lua')
torch.include('dp', 'hyper/priorsampler.lua')
torch.include('dp', 'hyper/imageclassfactory.lua')
torch.include('dp', 'hyper/mlpfactory.lua')
torch.include('dp', 'hyper/contextwordfactory.lua')
torch.include('dp', 'hyper/lmfactory.lua')

--[[ postgres ]]--
torch.include('dp', 'postgres/postgres.lua')
torch.include('dp', 'postgres/logger.lua')
torch.include('dp', 'postgres/xplog.lua')
torch.include('dp', 'postgres/savetofile.lua')
torch.include('dp', 'postgres/earlystopper.lua')
torch.include('dp', 'postgres/done.lua')
torch.include('dp', 'postgres/mlpfactory.lua')
torch.include('dp', 'postgres/lmfactory.lua')
torch.include('dp', 'postgres/hyperoptimizer.lua')

--[[ test ]]--
torch.include('dp', 'test/test.lua')
torch.include('dp', 'test/test-cuda.lua')
torch.include('dp', 'test/test-datasets.lua')

return dp
