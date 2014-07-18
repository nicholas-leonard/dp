------------------------------------------------------------------------
--[[ LMFactory ]]--
-- An experiment builder for training BillionWords using a
-- neural network language model (LM) of arbitrary dept
------------------------------------------------------------------------
local LMFactory, parent = torch.class("dp.LMFactory", "dp.MLPFactory")
LMFactory.isLMFactory = true
   
function LMFactory:__init(config)
   config = config or {}
   local args, name, logger, save_strategy = xlua.unpack(
      {config},
      'LMFactory', 
      'An experiment builder for training BillionWords using a '..
      'neural network language model (LM) of arbitrary dept',
      {arg='name', type='string', default='LM'},
      {arg='logger', type='dp.Logger', 
       help='defaults to dp.FileLogger'},
      {arg='save_strategy', type='object', 
       help='defaults to dp.SaveToFile()'}
   )
   config.name = name
   self._save_strategy = save_strategy or dp.SaveToFile()
   parent.__init(self, config)
   self._logger = logger or dp.FileLogger()
end

function LMFactory:addInput(mlp, activation, input_size, opt)
   mlp:add(
      dp.Dictionary{
         dict_size = opt.vocabularySize,
         output_size = opt.inputEmbeddingSize,
         acc_update = opt.accUpdate
      }
   )
   local input_size = opt.contextSize*opt.inputEmbeddingSize
   print("Input to first hidden layer has ".. input_size.." neurons.")
   return input_size
end

function LMFactory:addOutput(mlp, input_size, opt)
   mlp:add(
      dp.Neural{
         input_size=input_size, output_size=opt.outputEmbeddingSize,
         transfer=self:buildTransfer(opt.activation), 
         dropout=self:buildDropout(opt.dropout_probs[#(opt.dropout_probs)-1]),
         acc_update=opt.accUpdate
      }
   )
   print(opt.nClasses.." output neurons")
   local rootId = opt.datasource:rootId()
   local softmax
   if opt.softmaxforest then
      softmax = dp.SoftmaxForest{
         input_size = opt.outputEmbeddingSize, 
         hierarchy = {  
            opt.datasource:hierarchy('word_tree1.th7'), 
            opt.datasource:hierarchy('word_tree2.th7'),
            opt.datasource:hierarchy('word_tree3.th7')
         },
         gater_size = table.fromString(opt.forestGaterSize),
         gater_act = self:buildTransfer(opt.activation),
         root_id = {rootId,rootId,rootId},
         dropout = self:buildDropout(opt.dropout_probs[#(opt.dropout_probs)]),
         acc_update = opt.accUpdate
      }
      opt.softmaxtree = true
   elseif opt.softmaxtree then
      softmax = dp.SoftmaxTree{
         input_size = opt.outputEmbeddingSize, 
         hierarchy = datasource:hierarchy(),
         root_id = rootId,
         dropout = self:buildDropout(opt.dropout_probs[#(opt.dropout_probs)]),
         acc_update = opt.accUpdate
      }
   else
      print("Warning: you are using full LogSoftMax for last layer, which "..
         "is really slow (800,000 x outputEmbeddingSize multiply adds "..
         "per example. Try --softmaxtree instead.")
      softmax = dp.Neural{
         input_size = opt.outputEmbeddingSize,
         output_size = opt.nClasses,
         transfer = nn.LogSoftMax(),
         dropout = self:buildDropout(opt.dropout_probs[#(opt.dropout_probs)]),
         acc_update = opt.accUpdate
      }
   end
   print(opt.nClasses.." output neurons")
end

function LMFactory:buildModel(opt)
   if opt.softmaxtree or opt.softmaxforest then
      require 'cunnx'
   end
   --[[Model]]--
   local mlp = dp.Sequential()
   -- input layer
   local input_size = self:addInput(mlp, nil, nil, opt)
   -- hidden layer(s)
   local last_size = self:addHidden(mlp, opt.activation, input_size, 1, opt)
   -- output layer
   self:addOutput(mlp, last_size, opt)
   --[[GPU or CPU]]--
   if opt.model_type == 'cuda' then
      require 'cutorch'
      require 'cunn'
      mlp:cuda()
   elseif opt.model_type == 'double' then
      mlp:double()
   elseif opt.model_type == 'float' then
      mlp:float()
   end
   return mlp
end

function LMFactory:buildVisitor(opt)
   --[[ Visitor ]]--
   local visitor = {}
   table.insert(visitor, 
      dp.Learn{
         learning_rate = opt.learning_rate, 
         observer = self:buildLearningRateSchedule(opt)
      }
   )
   if opt.max_out_norm and opt.max_out_norm > 0 then
      table.insert(visitor, dp.MaxNorm{max_out_norm=opt.max_out_norm})
   end
   return visitor
end

function LMFactory:buildOptimizer(opt)
   --[[Propagators]]--
   return dp.Optimizer{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      visitor = self:buildVisitor(opt),
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{ --we assume large datasets (no shuffling)
         epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
      },
      progress = opt.progress or true
   }
end

function LMFactory:buildValidator(opt)
   return dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{
         epoch_size = opt.validEpochSize, 
         batch_size = opt.softmaxtree and 1024 or opt.batchSize
      }
   }
end

function LMFactory:buildTester(opt)
   return dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{
         batch_size = opt.softmaxtree and 1024 or opt.batchSize
      }
   }
end

function LMFactory:buildObserver(opt)
   return {
      self._logger,
      dp.EarlyStopper{
         start_epoch = 11,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10
      }
   }
end
