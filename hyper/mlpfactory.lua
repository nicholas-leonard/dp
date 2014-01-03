------------------------------------------------------------------------
--[[ MLPFactory ]]--
-- An example experiment builder for training Mnist using an 
-- MLP of arbitrary dept
------------------------------------------------------------------------
local MLPFactory, parent = torch.class("dp.MLPFactory", "dp.ExperimentFactory")
MLPFactory.isMLPFactory = true
   
function MLPFactory:__init(config)
   config = config or {}
   local args, name, logger, save_strategy = xlua.unpack(
      {config},
      'MLPFactory', nil,
      {arg='name', type='string', default='MLP'},
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

function MLPFactory:buildTransfer(activation)
   if activation == 'ReLU' then
      require 'nnx'
      return nn.ReLU()
   elseif activation == 'Tanh' then
      return nn.Tanh()
   elseif activation == 'Sigmoid' then
      return nn.Sigmoid()
   elseif activation ~= 'Linear' then
      error("Unknown activation function : " .. activation)
   end
end

function MLPFactory:buildDropout(dropout_prob)
   if dropout_prob then
      require 'nnx'
      return nn.Dropout(dropout_prob)
   end
end

function MLPFactory:buildModel(opt)
   local function addHidden(mlp, activation, input_size, layer_index)
      layer_index = layer_index or 1
      local output_size = opt.model_width * opt.width_scales[layer_index]
      mlp:add(
         dp.Neural{
            input_size=input_size, output_size=output_size,
            transfer=self:buildTransfer(activation), 
            dropout=self:buildDropout(opt.dropout_probs[layer_index])
         }
      )
      if layer_index < (opt.model_dept-1) then
         return addHidden(mlp, activation, output_size, layer_index+1)
      else
         return output_size
      end
   end
   --[[Model]]--
   local mlp = dp.Sequential()
   -- hidden layer(s)
   local last_size = addHidden(mlp, opt.activation, opt.feature_size, 1)
   -- output layer
   mlp:add(
      dp.Neural{
         input_size=last_size, output_size=#opt.classes,
         transfer=nn.LogSoftMax(), 
         dropout=self:buildDropout(opt.dropout_probs[layer_index])
      }
   )
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

function MLPFactory:buildLearningRateSchedule(opt)
   --[[ Schedules ]]--
   local lr = opt.learning_rate
   if opt.learning_decay1 ~= 'none' then
      local schedule = {[opt.learning_decay1] = 0.1 * lr}
      if opt.learning_decay2 ~= 'none' then
         schedule[opt.learning_decay2 + opt.learning_decay1] = 0.01 * lr
      end
      return dp.LearningRateSchedule{schedule=schedule}
   end
end

function MLPFactory:buildVisitor(opt)
   local lr_schedule = self:buildLearningRateSchedule(opt)
   --[[ Visitor ]]--
   local visitor = {}
   if opt.momentum and opt.momentum ~= 0 then
      table.insert(visitor, 
         dp.Momentum{
            momentum_factor=opt.momentum, 
            nesterov=opt.nesterov
         }
      )
   end
   if opt.weightdecay and opt.weightdecay ~=0 then
      table.insert(visitor, dp.WeightDecay{wd_factor=opt.weightdecay})
   end
   table.insert(visitor, 
      dp.Learn{
         learning_rate = opt.learning_rate, 
         observer = lr_schedule
      }
   )
   if opt.max_out_norm and opt.max_out_norm ~= 0 then
      table.insert(visitor, dp.MaxNorm{max_out_norm=opt.max_out_norm})
   end
   return visitor
end

function MLPFactory:buildOptimizer(opt)
   local visitor = self:buildVisitor(opt)
   --[[Propagators]]--
   return dp.Optimizer{
      criterion = nn.ClassNLLCriterion(),
      visitor = visitor,
      feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{
         batch_size=opt.batch_size, 
         sample_type=opt.model_type
      },
      progress = true
   }
end

function MLPFactory:buildValidator(opt)
   return dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{sample_type=opt.model_type}
   }
end

function MLPFactory:buildTester(opt)
   return dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),
      sampler = dp.Sampler{sample_type=opt.model_type}
   }
end

function MLPFactory:buildObserver(opt)
   return {
      self._logger,
      dp.EarlyStopper{
         start_epoch = 1,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10, max_error = 70
      }
   }
end

function MLPFactory:build(opt, id)
   --[[Experiment]]--
   return dp.Experiment{
      id = id,
      random_seed = opt.random_seed,
      model = self:buildModel(opt),
      optimizer = self:buildOptimizer(opt),
      validator = self:buildValidator(opt),
      tester = self:buildTester(opt),
      observer = self:buildObserver(opt),
      max_epoch = opt.max_epoch
   }
end
