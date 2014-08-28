require 'dp'
require 'cunnx'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Language Model on BillionWords dataset using a serialized experiment')
cmd:text('Example:')
cmd:text('$> th trainserialized.lua --xpFile nps:26693:1408205007:1 ')
cmd:text('$> th trainserialized.lua --xpFile nps:26693:1408205007:1.dat --learningRate 0.1 --decayPoints "{10,75,90,100,125}" --learningRates "{0.2,0.075,0.05,0.025,0.01}"')
cmd:text('Options:')
cmd:option('--xpDir', dp.SAVE_DIR, 'directory containing experiment')
cmd:option('--xpFile', '', 'name of file containing experiment')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')

cmd:option('--learningRate', -1, 'new learning rate of model')
cmd:option('--newSchedule', false, 'setup new learning rate decay schedule')
cmd:option('--decayPoints', '{20,100,125}', 'epochs at which learning rate is decayed')
cmd:option('--learningRates', '{0.2,0.1,0.01}', 'learning rate at decay points')
cmd:option('--batchSize', -1, 'new batchSize of training set')

cmd:option('--contextSize', 5, 'number of words preceding the target word used to predict the target work')
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

cutorch.setDevice(opt.useDevice) --will the experiment be loaded here?
assert(xpFile ~= '', "missing xpFile argument")

xp = torch.load(paths.concat(opt.xpDir, opt.xpFile))

--[[data]]--
local train_file = 'train_data.th7' 
if opt.small then 
   train_file = 'train_small.th7'
elseif opt.tiny then 
   train_file = 'train_tiny.th7'
end
local datasource = dp.BillionWords{
   context_size = opt.contextSize, train_file = train_file
}

if opt.learningRate > 0 then
   xp:optimizer():visitor()._visitors[1]._learning_rate = opt.learningRate
end
if opt.newSchedule then
   opt.decayPoints = table.fromString(opt.decayPoints)
opt.learningRates = table.fromString(opt.learningRates)
   local schedule = {}
   for i,decayPoint in ipairs(opt.decayPoints) do
      schedule[decayPoint] = opt.learningRates[i]
   end
   print("schedule:", schedule)
   xp:optimizer():visitor()._visitors[1]._observer._schedule = schedule
end

if opt.batchSize > 0 then
   xp:optimizer():sampler():setBatchSize(opt.batchSize)
end

xp:run(datasource)
