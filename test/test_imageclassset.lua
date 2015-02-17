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

ppf = ds:normalizePPF()

trainSet = ds:trainSet()
start = os.clock()
for i=1,100 do
   trainSet:sample(batch,120,'sampleTrain') 
   collectgarbage()
end
print("sampleTrain", (os.clock()-start)/100)

start = os.clock()
for i=1,12*100,12 do
   trainSet:sub(batch,i,i+11) 
   collectgarbage()
end
print("sampleTest", (os.clock()-start)/100)

trainSet = dp.SVHN():trainSet()
start = os.clock()
for i=1,120*100,120 do
   trainSet:sub(batch,i,i+119) 
   collectgarbage()
end
print("SVHN", (os.clock()-start)/100)
