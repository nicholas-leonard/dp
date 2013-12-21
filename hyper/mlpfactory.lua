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

function MLPFactory:buildModel(opt)
   local function addHidden(mlp, activation, input_size, layer_index)
      layer_index = layer_index or 1
      local output_size = opt.model_width * opt.width_scales[layer_index]
      local transfer
      if activation == 'ReLU' then
         require 'nnx'
         transfer = nn.ReLU()
      elseif activation == 'Tanh' then
         transfer = nn.Tanh()
      elseif activation == 'Sigmoid' then
         transfer = nn.Sigmoid()
      elseif activation ~= 'Linear' then
         error("Unknown activation function : " .. activation)
      end
      local dropout
      if opt.dropout_probs[layer_index] then
         require 'nnx'
         dropout = nn.Dropout(opt.dropout_probs[layer_index])
      end
      mlp:add(dp.Neural{input_size=input_size, output_size=output_size,
                        transfer=transfer, dropout=dropout})
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
   local dropout
   if opt.dropout_probs[opt.model_dept] then
      require 'nnx'
      dropout = nn.Dropout(opt.dropout_probs[opt.model_dept])
   end
   mlp:add(
      dp.Neural{
         input_size=last_size, output_size=#opt.classes,
         transfer=nn.LogSoftMax(), dropout=dropout
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

--TODO : make this more flexible : sample first epoch, then delta to 2nd
function MLPFactory:buildLearningRateSchedule(opt)
   --[[ Schedules ]]--
   local schedule
   local lr = opt.learning_rate
   if opt.learning_schedule == '100=/10,200=/10' then 
      schedule={[100]=0.1*lr, [200]=0.01*lr}
   elseif opt.learning_schedule == '100=/10,150=/10' then
      schedule={[100]=0.1*lr, [150]=0.01*lr}
   elseif opt.learning_schedule == '200=/10,250=/10' then
      schedule={[200]=0.1*lr, [250]=0.01*lr}
   elseif opt.learning_schedule == 'none' then
      -- pass
   elseif opt.learning_schedule then
      error("Unknown learning schedule : " .. opt.learning_schedule)
   end
   local lr_schedule 
   if opt.learning_schedule and opt.learning_schedule ~= 'none' then 
      lr_schedule = dp.LearningRateSchedule{schedule=schedule}
   end
   return lr_schedule
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
         save_strategy = self._save_strategy
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
