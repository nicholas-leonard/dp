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
