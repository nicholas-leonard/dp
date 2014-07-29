require 'dp'

--[[parse command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL Language Model Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pglm.lua --collection "BW-LM-1" --batchSize 128 --momentum 0.5')
cmd:text('$> th pglm.lua --collection "BW-LM-2" --batchSize 128 --learningRate 0.1 --modelWidth 1024 --widthScales "{1,0.37109375,0.043945312}" --modelDept 4 --progress')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoints', '{400,600,700}', 'epochs at which learning rate is decayed')
cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed at each point')
cmd:option('--linearDecay', false, 'linear decay from first to second from second to third point, etc')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 2, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--modelWidth', 1024, 'width of the model in hidden neurons')
cmd:option('--widthScales', '{1,1,1}', 'scales the width of different layers')
cmd:option('--modelDept', 2, 'number of Neural layers (affine transform followed by transfer function) to use')
cmd:option('--activation', 'Tanh', 'activation function')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropoutProbs', '{0}', 'probability of dropout on inputs to each layer')
cmd:option('--collection', 'lm-bw-1', 'identifies a collection of related experiments')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--pg', false, 'use postgresql')
cmd:option('--maxError', 0, 'maximum NLL that must be maintained after 10 epochs (0 means ignore)')
cmd:option('--datasource', 'BillionWords', 'datasource to use : Mnist | NotMnist | Cifar10')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')

cmd:option('--contextSize', 5, 'number of words preceding the target word used to predict the target work')
cmd:option('--inputEmbeddingSize', 100, 'number of neurons per word embedding')
cmd:option('--outputEmbeddingSize', 100, 'number of hidden units at softmaxtree')
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('--softmaxforest', false, 'use SoftmaxForest instead of SoftmaxTree (uses more memory)')
cmd:option('--forestGaterSize', '{}', 'size of hidden layers used for forest gater (trees are experts)')
cmd:option('--small', false, 'use a small (1/30th) subset of the BillionWords training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the BillionWords training set')
cmd:option('--trainEpochSize', 1000000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 100000, 'number of valid examples used for early stopping and cross-validation') 
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

local hp = {
   version = 1,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   tiny = opt.tiny,
   small = opt.small,
   train_epoch_size = opt.trainEpochSize,
   valid_epoch_size = opt.validEpochSize,
   context_size = opt.contextSize,
   input_embedding_size = opt.inputEmbeddingSize,
   output_embedding_size = opt.outputEmbeddingSize,
   softmaxtree = opt.softmaxtree or opt.softmaxforest,
   softmaxforest = opt.softmaxforest,
   forest_gater_size = table.fromString(opt.forestGaterSize),
   datasource = opt.datasource,
   random_seed = dp.TimeChoose(),
   batch_size = opt.batchSize,
   model_dept = opt.modelDept,
   learning_rate = opt.learningRate,
   decay_points = table.fromString(opt.decayPoints),
   decay_factor = opt.decayFactor,
   linear_decay = opt.linearDecay,
   max_out_norm = opt.maxOutNorm,
   max_norm_period = opt.maxNormPeriod,
   model_width = opt.modelWidth,
   width_scales = table.fromString(opt.widthScales),
   activation = opt.activation,
   dropout_probs = table.fromString(opt.dropoutProbs),
   collection = opt.collection,
   progress = opt.progress,
   max_error = opt.minAccuracy,
   acc_update = opt.accUpdate
}

if not opt.pg then
   local logger = dp.FileLogger()
   hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name='LM+'..opt.datasource..':user_dist', dist=hp 
      },
      experiment_factory = dp.LMFactory{
         logger=logger,
         save_strategy=dp.SaveToFile()
      },
      datasource_factory=dp.ContextWordFactory(),
      logger=logger
   }
   hyperopt:run()
end

local pg = dp.Postgres()
local logger = dp.PGLogger{pg=pg}

hyperopt = dp.PGHyperOptimizer{
   collection_name=opt.collection,
   hyperparam_sampler = dp.PriorSampler{--only samples random_seed
      name='LM+'..opt.datasource..':user_dist', dist=hp 
   },
   experiment_factory = dp.PGLMFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{pg=pg}
   },
   datasource_factory=dp.ContextWordFactory(),
   logger=logger
}

hyperopt:run()
