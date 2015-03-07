require 'dp'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MLP Hyperparameter Optimization using a prior distribution')
cmd:text('Example:')
cmd:text('$> th hyperoptimize.lua --maxEpoch 500 --maxTries 50 --collection "MnistMLP1"')
cmd:text('Options:')
cmd:option('--collection', 'hyperoptimization_example_1', 'identifies a collection of related experiments (no spaces)')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use for this hyperoptimization')
cmd:option('--datasource', 'Mnist', 'datasource to use : Mnist | NotMnist | Cifar10')
cmd:option('--batchSize', -1, 'number of examples per batch')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoints', '{400,600,700}', 'epochs at which learning rate is decayed')
cmd:option('--decayFactor', -1, 'factor by which learning rate is decayed at each point e.g. 0.1')
cmd:option('--linearDecay', false, 'linear decay from first to second from second to third point, etc')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights e.g. 1')
cmd:option('--weightDecay', -1, 'weight decay factor')
cmd:option('--momentum', -1, 'momentum')
cmd:option('--nesterov', false, 'use nesterov momentum')
cmd:option('--modelWidth', -1, 'width of the model in hidden neurons. e.g. 1024')
cmd:option('--widthScales', '', 'scales the width of different layers e.g. "{1,1,1}"')
cmd:option('--modelDept', -1, 'number of Neural layers (affine transform followed by transfer function) to use e.g. 2')
cmd:option('--activation', '', 'activation function. e.g. "Tanh"')
cmd:option('--dropoutProbs', '{0}', 'probability of dropout on inputs to each layer')

cmd:option('--zca_gcn', false, 'apply GCN followed by ZCA input preprocessing')
cmd:option('--standardize', false, 'apply Standardize input preprocessing')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing to datasource inputs')
cmd:option('--validRatio', 1/6, 'proportion of train set used for cross-validation')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--pg', false, 'use postgresql')
cmd:option('--minAccuracy', 0.1, 'minimum accuracy that must be maintained after 10 epochs')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

dist = {
   version = 1,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = opt.datasource,
   random_seed = dp.TimeChoose(),
   decay_points = table.fromString(opt.decayPoints),
   linear_decay = opt.linearDecay,
   nesterov = opt.nesterov,
   valid_ratio = opt.validRatio,
   collection = opt.collection,
   progress = opt.progress,
   zca_gcn = opt.zca_gcn,
   standardize = opt.standardize,
   lecunlcn = opt.lecunLCN,
   max_error = opt.minAccuracy,
   batch_size = (opt.batchSize == -1) and dp.WeightedChoose{
      [32]=10, [64]=7, [128]=5, [256]=4, [16]=3, [8]=2, [512]=1 
   } or opt.batchSize,
   learning_rate = (opt.learningRate == -1) and dp.WeightedChoose{
      [0.5]=0.1, [0.1]=0.8, [0.05]=0.1, [0.01]=0.3, [0.001]=0.1
   } or opt.learningRate,
   decay_factor = (opt.decayFactor == -1) and dp.WeightedChoose{
      [0.3]=5, [0.2]=4, [0.1]=3
   } or opt.decayFactor,
   max_out_norm = (opt.maxOutNorm == -1) and dp.WeightedChoose{
      [0]=1, [0.5]=1, [1]=10, [2]=2 , [4]=1
   } or opt.maxOutNorm,
   weight_decay = (opt.weightDecay == -1) and dp.WeightedChoose{
      [0.0005] = 0.1, [0.00005] = 0.7, [0.000005] = 0.2, [0]=10
   } or opt.weightDecay,
   momentum = (opt.momentum == -1) and dp.WeightedChoose{
      [0] = 1, [0.5] = 0.1, [0.7] = 0.1, [0.9] = 0.3, [0.99] = 0.5
   } or opt.momentum,
   model_dept = (opt.modelDept == -1) and dp.WeightedChoose{
      [2] = 0.9, [3] = 0.05, [4] = 0.05
   } or opt.modelDept,
   model_width = (opt.modelWidth == -1) and dp.WeightedChoose{
      [128]=0.1, [256]=0.2, [512]=0.3, [1024]=0.3, [2048]=0.1
   } or opt.modelWidth,
   width_scales = (opt.widthScales == '') and dp.WeightedChoose{
      [{1,1,1}]=0.5,      [{1,0.5,0.5}]=0.1, [{1,1,0.5}]=0.1,
      [{1,0.5,0.25}]=0.1, [{0.5,1,0.5}]=0.1, [{1,0.25,0.25}]=0.1
   } or table.fromString(opt.widthScales),
   activation = (opt.activation == '') and dp.WeightedChoose{
      ['Tanh'] = 0.4, ['ReLU'] = 0.5, ['Sigmoid'] = 0.1
   } or opt.activation,
   dropout_probs = (opt.dropoutProbs == '') and dp.WeightedChoose{
      [{false,false,false,false}] = 0.4, 
      [{0.2,0.5,0.5,0.5}] = 0.2, 
      [{false,0.5,0.5,0.5}] = 0.5
   } or table.fromString(opt.dropoutProbs)
}

local hyperopt
if opt.pg then
   local pg = dp.Postgres()
   local logger = dp.PGLogger{pg=pg}
   
   hyperopt = dp.PGHyperOptimizer{
      collection_name=opt.collection,
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name='MLP+'..opt.datasource..':user_dist', dist=dist
      },
      experiment_factory = dp.PGMLPFactory{
         logger=logger, pg=pg, 
         save_strategy=dp.PGSaveToFile{pg=pg}
      },
      datasource_factory=dp.ImageClassFactory(),
      logger=logger
   }
else
   local logger = dp.FileLogger()
   
   hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name='MLP+'..opt.datasource..':user_dist', dist=dist
      },
      experiment_factory = dp.MLPFactory{
         logger=logger,
         save_strategy=dp.SaveToFile()
      },
      datasource_factory=dp.ImageClassFactory(),
      logger=logger
   }
end

hyperopt:run()
