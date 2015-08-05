require 'dp'
require 'rnn'

version = 9

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords or PennTreeBank (or your own) dataset using a Simple Recurrent Neural Network')
cmd:text('Example:')
cmd:text("$> th recurrentlanguagemodel.lua --dataset PennTreeBank --cuda --useDevice 2 --trainEpochSize -1 --trainEpochSize -1 --dropout --bidirectional --hiddenSize '{200,200}' --zeroFirst --batchSize 32 --progress")
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 64 ')
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 64 --rho 5 --validEpochSize 10000 --trainEpochSize 100000 --softmaxtree')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--maxOutNorm', 2, 'max l2-norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--evalSize', 100, 'size of context used for evaluation (more means more memory). With --bidirectional, specifies number of steps between each bwd rnn forget() (more means longer bwd recursions)')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--uniform', -1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

--[[ recurrent layer ]]--
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--bidirectional', false, 'use a Bidirectional RNN/LSTM (nn.BiSequencer instead of nn.Sequencer)')
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')


--[[ output layer ]]--
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('--softmaxforest', false, 'use SoftmaxForest instead of SoftmaxTree (uses more memory)')
cmd:option('--forestGaterSize', '{}', 'size of hidden layers used for forest gater (trees are experts)') 

--[[ data ]]--
cmd:option('--dataset', 'BillionWords', 'which dataset to use : BillionWords | PennTreeBank | TextSource')
cmd:option('--trainEpochSize', 400000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 24000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--dataPath', dp.DATA_DIR,  'path to data directory')
-- BillionWords
cmd:option('--small', false, 'use a small (1/30th) subset of the training set (BillionWors only)')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set (BillionWors only)')
-- TextSource
cmd:option('--trainFile', 'train.txt', 'filename containing tokenized training text data')
cmd:option('--validFile', 'valid.txt', 'filename containing tokenized validation text data')
cmd:option('--testFile', 'test.txt', 'filename containing tokenized test text data')

cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

if opt.bidirectional and not opt.silent then
   print("Warning : the Perplexity of a bidirectional RNN/LSTM isn't "..
      "necessarily mathematically valid as it uses P(x_t|x_{/neq t}) "..
      "instead of P(x_t|x_{<t}), which is used for unidirectional RNN/LSTMs. "..
      "You can however still use predictions to measure pseudo-likelihood.")
end

if opt.xpPath ~= '' then
   -- check that saved model exists
   assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

--[[Data]]--
local train_file = 'train_data.th7' 
if opt.small then 
   train_file = 'train_small.th7'
elseif opt.tiny then 
   train_file = 'train_tiny.th7'
end

if opt.dataset == 'BillionWords' then
   assert(not opt.bidirectional, "--bidirectional not yet supported with BillionWords")
   ds = dp.BillionWords{
      train_file=train_file, load_all=false, 
      context_size=opt.rho, recurrent=true
   }
   ds:loadTrain()
   if not opt.trainOnly then
      ds:loadValid()
      ds:loadTest()
   end
elseif opt.dataset == 'PennTreeBank' or opt.dataset == 'TextSource' then
   assert(not opt.softmaxforest, "SoftMaxForest only supported with BillionWords")
   if opt.dataset == 'PennTreeBank' then
      ds = dp.PennTreeBank{
         context_size=opt.bidirectional and opt.rho+1 or opt.rho, 
         recurrent=true, bidirectional=opt.bidirectional
      }
   elseif opt.dataset == 'TextSource' then
      ds = dp.TextSource{
         context_size=opt.bidirectional and opt.rho+1 or opt.rho, 
         recurrent=true, bidirectional=opt.bidirectional,
         name='rnnlm', data_path = opt.dataPath,
         train=opt.trainFile, valid=opt.validFile, test=opt.testFile
      }
   end
   ds:validSet():contextSize(opt.evalSize)
   ds:testSet():contextSize(opt.evalSize)
else
   error"Unrecognized --dataset"
end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'cunnx'
      cutorch.setDevice(opt.useDevice)
   end
   xp = torch.load(opt.xpPath)
   if opt.cuda then
      xp:cuda()
   else
      xp:float()
   end
   xp:run(ds)
   os.exit()
end

--[[Model]]--

-- language model
lm = nn.Sequential()

local inputSize = opt.hiddenSize[1]
for i,hiddenSize in ipairs(opt.hiddenSize) do 

   if i~= 1 and not opt.lstm then
      lm:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
   end
   
   -- recurrent layer
   local rnn
   if opt.lstm then
      -- Long Short Term Memory
      rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
   else
      -- simple recurrent neural network
      rnn = nn.Recurrent(
         hiddenSize, -- first step will use nn.Add
         nn.Identity(), -- for efficiency (see above input layer) 
         nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
         nn.Sigmoid(), -- transfer function 
         99999 -- maximum number of time-steps per sequence
      )
      if opt.zeroFirst then
         -- this is equivalent to forwarding a zero vector through the feedback layer
         rnn.startModule:share(rnn.feedbackModule, 'bias')
      end
      rnn = nn.Sequencer(rnn)
   end

   lm:add(rnn)
   
   if opt.dropout then -- dropout it applied between recurrent layers
      lm:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hiddenSize
end

if opt.bidirectional then
   -- initialize BRNN with fwd, bwd RNN/LSTMs
   local bwd = lm:clone()
   bwd:reset()
   bwd:remember('neither')
   local brnn = nn.BiSequencerLM(lm, bwd)
   
   lm = nn.Sequential()
   lm:add(brnn)
   
   inputSize = inputSize*2
end

-- input layer (i.e. word embedding space)
lm:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors

if opt.dropout then
   lm:insert(nn.Dropout(opt.dropoutProb), 1)
end

lookup = nn.LookupTable(ds:vocabularySize(), opt.hiddenSize[1], opt.accUpdate)
lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
lm:insert(lookup, 1)

-- output layer
if opt.softmaxforest or opt.softmaxtree then
   -- input to nnlm is {inputs, targets} for nn.SoftMaxTree
   local para = nn.ParallelTable()
   para:add(lm):add(opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity())
   lm = nn.Sequential()
   lm:add(para)
   lm:add(nn.ZipTable())
   if opt.softmaxforest then -- requires a lot more memory
      local trees = {ds:hierarchy('word_tree1.th7'), ds:hierarchy('word_tree2.th7'), ds:hierarchy('word_tree3.th7')}
      local rootIds = {880542,880542,880542}
      softmax = nn.SoftMaxForest(inputSize, trees, rootIds, opt.forestGaterSize, nn.Tanh(), opt.accUpdate)
      opt.softmaxtree = true
   elseif opt.softmaxtree then -- uses frequency based tree
      local tree, root = ds:frequencyTree()
      softmax = nn.SoftMaxTree(inputSize, tree, root, opt.accUpdate)
   end
else
   if #ds:vocabulary() > 50000 then
      print("Warning: you are using full LogSoftMax for last layer, which "..
         "is really slow (800,000 x outputEmbeddingSize multiply adds "..
         "per example. Try --softmaxtree instead.")
   end
   softmax = nn.Sequential()
   softmax:add(nn.Linear(inputSize, ds:vocabularySize()))
   softmax:add(nn.LogSoftMax())
end
lm:add(nn.Sequencer(softmax))

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

if opt.dataset ~= 'BillionWords' then
   -- will recurse a single continuous sequence
   lm:remember(opt.lstm and 'both' or 'eval')
end
   

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
   loss = opt.softmaxtree and nn.SequencerCriterion(nn.TreeNLLCriterion())
         or nn.ModuleCriterion(
            nn.SequencerCriterion(nn.ClassNLLCriterion()), 
            nn.Identity(), 
            opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity()
         ),
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
            if opt.meanNorm then
               print("mean gradParam norm", opt.meanNorm)
            end
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         if opt.cutoffNorm > 0 then
            local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
            opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         end
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Perplexity(),  
   sampler = torch.isTypeOf(ds, 'dp.TextSource')
      and dp.TextSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}
      or dp.RandomSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}, 
   acc_update = opt.accUpdate,
   progress = opt.progress
}

if not opt.trainOnly then
   valid = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') 
         and dp.TextSampler{epoch_size = opt.validEpochSize, batch_size = 1} 
         or dp.SentenceSampler{epoch_size = opt.validEpochSize, batch_size = 1, max_size = 100},
      progress = opt.progress
   }
   tester = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') 
         and dp.TextSampler{batch_size = 1} 
         or dp.SentenceSampler{batch_size = 1, max_size = 100}  -- Note : remove max_size for exact test set perplexity (will cost more memory)
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = lm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','perplexity','ppl'}
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch,
   target_module = nn.SplitTable(1,1):type('torch.IntTensor')
}
if opt.softmaxtree then
   -- makes it forward {input, target} instead of just input
   xp:includeTarget()
end

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   if opt.softmaxtree or opt.softmaxforest then
      require 'cunnx'
   end
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Language Model :"
   print(lm)
end

xp:run(ds)
