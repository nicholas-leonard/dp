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
   if dropout_prob and dropout_prob > 0 and dropout_prob < 1 then
      require 'nnx'
      return nn.Dropout(dropout_prob)
   end
end

function MLPFactory:buildModel(opt)
   local function addHidden(mlp, activation, input_size, layer_index)
      layer_index = layer_index or 1
      local output_size = math.ceil(
         opt.model_width * opt.width_scales[layer_index]
      )
      mlp:add(
         dp.Neural{
            input_size=input_size, output_size=output_size,
            transfer=self:buildTransfer(activation), 
            dropout=self:buildDropout(opt.dropout_probs[layer_index])
         }
      )
      print(output_size .. " hidden neurons")
      if layer_index < (opt.model_dept-1) then
         return addHidden(mlp, activation, output_size, layer_index+1)
      else
         return output_size
      end
   end
   --[[Model]]--
   local mlp = dp.Sequential()
   -- hidden layer(s)
   print(opt.feature_size .. " input neurons")
   local last_size = addHidden(mlp, opt.activation, opt.feature_size, 1)
   -- output layer
   mlp:add(
      dp.Neural{
         input_size=last_size, output_size=#opt.classes,
         transfer=nn.LogSoftMax(), 
         dropout=self:buildDropout(opt.dropout_probs[layer_index])
      }
   )
   print(#opt.classes.." output neurons")
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
   local start_lr = opt.learning_rate
   local schedule
   if opt.linear_decay then
      x = torch.range(1,opt.decay_points[#opt.decay_points])
      y = torch.FloatTensor(x:size()):fill(start_lr)
      for i = 2, #opt.decay_points do
         local start_epoch = opt.decay_points[i-1]
         local end_epoch = opt.decay_points[i]
         local end_lr = start_lr * opt.decay_factor
         local m = (end_lr - start_lr) / (end_epoch - start_epoch)
         y[{{start_epoch,end_epoch}}] = torch.mul(
            torch.add(x[{{start_epoch,end_epoch}}], -start_epoch), m
         ):add(start_lr)
         start_lr = end_lr
      end
      schedule = y
   else
      schedule = {}
      for i, epoch in ipairs(opt.decay_points) do
         start_lr = start_lr * opt.decay_factor
         schedule[epoch] = start_lr
      end
   end
   return dp.LearningRateSchedule{schedule=schedule}
end

function MLPFactory:buildVisitor(opt)
   local lr_schedule = self:buildLearningRateSchedule(opt)
   --[[ Visitor ]]--
   local visitor = {}
   if opt.momentum and opt.momentum > 0 then
      table.insert(visitor, 
         dp.Momentum{
            momentum_factor=opt.momentum, nesterov=opt.nesterov,
            exclude=opt.exclude_momentum
         }
      )
   end
   if opt.weightdecay and opt.weightdecay > 0 then
      table.insert(visitor, dp.WeightDecay{wd_factor=opt.weightdecay})
   end
   table.insert(visitor, 
      dp.Learn{
         learning_rate = opt.learning_rate, 
         observer = lr_schedule
      }
   )
   if opt.max_out_norm and opt.max_out_norm > 0 then
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
         batch_size=opt.batch_size, sample_type=opt.model_type
      },
      progress = true
   }
end

function MLPFactory:buildValidator(opt)
   return dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function MLPFactory:buildTester(opt)
   return dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function MLPFactory:buildObserver(opt)
   return {
      self._logger,
      dp.EarlyStopper{
         start_epoch = 11,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10, max_error = opt.max_error or 0.1
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
