require 'dp'

ds = dp.ImageNet{
   train_path='/media/Nick/ImageNet/ILSVRC2012_img_train',
   valid_path='/media/Nick/ImageNet/ILSVRC2012_img_val',
   meta_path='/media/Nick/ImageNet/metadata',
   load_all = false
}

validSet = ds:loadValid()
batch = validSet:sample(128)
print(batch:inputs():view(), batch:inputs():input():size())

batch = validSet:sample(128, 'sampleTest')
print(batch:inputs():view(), batch:inputs():input():size(), batch:targets():input())

batch = validSet:sample(4, 'sampleTrain')
print(batch:inputs():view(), batch:inputs():input():size())

batch = validSet:sub(100, 200)
print("sub1", batch:inputs():view(), batch:inputs():input():size())

validSet:sub(batch, 200, 240)
print("sub2", batch:inputs():view(), batch:inputs():input():size())


--ppf = ds:normalizePPF()

trainSet = ds:trainSet() or ds:loadTrain()

batch = trainSet:sample(batch,120,'sampleTrain')

inputView = batch:inputs() 
inputs = inputView:forward('bchw')

targetView = batch:targets()
targets = targetView:forward('b')

savePath = paths.concat(dp.SAVE_DIR, 'imagenettest')
paths.mkdir(savePath)

for i=1,inputs:size(1) do
   image.save(paths.concat(savePath, 'sample'..i..'_'..targets[i]..'.png'), inputs[i])
end

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   local img = image.load(imgpath)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImageB", (a:time().real)/10)

local a = torch.Timer()
float = torch.FloatTensor()
dst = torch.FloatTensor()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   local out = image.load(imgpath)
   float:resize(out:size()):copy(out)
   dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
   image.scale(dst, float)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImageB+scale", (a:time().real)/10)

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   trainSet:loadImage(imgpath) 
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage", (a:time().real)/10)


local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   input = trainSet:loadImage(imgpath) 
   local out = input:toTensor('float','RGB','DHW', true)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage+", (a:time().real)/10)

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   input = trainSet:loadImage(imgpath) 
   local out = input:toTensor('float','RGB','DHW', true)
   dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
   image.scale(dst, out)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage+scale", (a:time().real)/10)

local a = torch.Timer()
for i=1,120 do
   local dst = trainSet:getImageBuffer(i)
   dst:resize(10, 3, trainSet._sample_size[3], trainSet._sample_size[2])
end
print("getImageBuffer : first pass", (a:time().real)/120)

local a = torch.Timer()
for j=1,10 do
   local inputTable = {}
   local targetTable = {} 
   for i=1,120 do 
      assert(batch)
      local imgpath = ffi.string(torch.data(trainSet.imagePath[j*120+i]))
      input = trainSet:loadImage(imgpath) 
      local out = input:toTensor('float','RGB','DHW', true)
      local dst = trainSet:getImageBuffer(i)
      dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
      image.scale(dst, out)
      table.insert(inputTable, dst)
      table.insert(targetTable, 1)  
   end
   local inputView = batch and batch:inputs() or dp.ImageView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()
   
   trainSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   
   assert(inputTensor:size(2) == 3)
   inputView:forward('bchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(trainSet._classes)
   batch:setInputs(inputView)
   batch:setTargets(targetView)  
   batch:carry():putObj('nSample', targetTensor:size(1))
   collectgarbage()
end
print("loadImage+scale+tableToTensor", (a:time().real)/10)

a = torch.Timer()
for i=1,10 do
   trainSet:sample(batch,120,'sampleDefault') 
   collectgarbage()
end
print("sampleDefault", (a:time().real)/10)

a = torch.Timer()
for i=1,10 do
   trainSet:sample(batch,120,'sampleTrain') 
   collectgarbage()
end
print("sampleTrain", (a:time().real)/10)

a = torch.Timer()
for i=1,120*10,120 do
   trainSet:sub(batch,i,i+119) 
   collectgarbage()
end
print("sampleTest", (a:time().real)/10)

