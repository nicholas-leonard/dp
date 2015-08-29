--[[!
Generating difficult MNIST data.
Translated, Cluttered MNIST samples.
--]]

require 'dp'
require 'image'

op = xlua.OptionParser('%prog [options]')
op:option{'--transHeight', action='store', dest='transHeight',
          help='Translated/Cluttered image width ',default=60}
op:option{'--transWidth', action='store', dest='transWidth',
          help='Translated/Cluttered image width ',default=60}
op:option{'--noOfTrans', action='store', dest='noOfTrans',
          help='Number of translation', default=5}
op:option{'--translate', action='store_true', dest='translate',
          help='If true Translate the digits else clutter.', default=false}
op:option{'--patchHeight', action='store', dest='patchHeight',
          help='Translated/Cluttered image height ',default=8}
op:option{'--patchWidth', action='store', dest='patchWidth',
          help='Translated/Cluttered image width ',default=8}
op:option{'--noOfClutters', action='store', dest='noOfClutters',
          help='No. of other images to use for generating cluttered image.',
          default=5}
op:option{'-o', '--outdir', action='store', dest='outdir',
          help='Output Directory', default=""}
op:option{'--saveImage', action='store_true',
          dest='saveImage', help='Save generated images to disk.',
          default=false}

-- Set default tensor type to float
torch.setdefaulttensortype("torch.FloatTensor")

-- Generate translated digits
function translate_digit(sample, transHeight, transWidth)
   local transHeight = transHeight or 60
   local transWidth = transWidth or 60

   -- Sample Image size
   local channels = sample:size(1)
   local height = sample:size(2)
   local width = sample:size(3)

   assert(transHeight>height and transWidth>width,
          "Translated height should be greater than input height.")

   local transSample = torch.zeros(channels, transHeight, transWidth)
   local placeHeightIndx = torch.random(1, transHeight-height+1)
   local placeWidthIndx = torch.random(1, transWidth-width+1)

   transSample[{{}, {placeHeightIndx, placeHeightIndx+height-1},
                    {placeWidthIndx, placeWidthIndx+width-1}}]:copy(sample)

   return transSample
end

-- Generate cluttered digits
function cluttered_digit(sample, otherSamples, noOfClutters,
                         transHeight, transWidth,
                         clutterPatchHeight, clutterPatchWidth)
   local noOfClutters = noOfClutters or 4
   local transHeight = transHeight or 60
   local transWidth = transWidth or 60
   local clutterPatchHeight = clutterPatchHeight or 8
   local clutterPatchWidth = clutterPatchWidth or 8

   -- Sample Image size
   local channels = sample:size(1)
   local height = sample:size(2)
   local width = sample:size(3)

   assert((transHeight>height and transWidth>width) and
          (clutterPatchHeight < height and clutterPatchWidth < width),
          "Translated height should be greater than input height.")

   local clutteredSample = torch.zeros(channels, transHeight, transWidth)
   local placeHeightIndx = torch.random(1, transHeight-height+1)
   local placeWidthIndx = torch.random(1, transWidth-width+1)

   clutteredSample[{{}, {placeHeightIndx, placeHeightIndx+height-1},
                   {placeWidthIndx, placeWidthIndx+width-1}}]:copy(sample)

   -- Add cluttering
   local noOfOtherSamples = otherSamples:size(1)
   for i=1, noOfClutters do
      local sampleIndx = torch.random(1, noOfOtherSamples)
      local patchHeightIndx = torch.random(1, height-clutterPatchHeight+1)
      local patchWidthIndx = torch.random(1, width-clutterPatchWidth+1)
      local clutterPatch = otherSamples[sampleIndx][{{},
                                       {patchHeightIndx,
                                        patchHeightIndx+clutterPatchHeight-1},
                                       {patchWidthIndx,
                                        patchWidthIndx+clutterPatchWidth-1}}]
      placeHeightIndx = torch.random(1, transHeight-clutterPatchHeight+1)
      placeWidthIndx = torch.random(1, transWidth-clutterPatchWidth+1)
      clutteredSample[{{}, {placeHeightIndx,
                            placeHeightIndx+clutterPatchHeight-1},
                      {placeWidthIndx,
                       placeWidthIndx+clutterPatchWidth-1}}]:copy(clutterPatch)
   end
   return clutteredSample
end

-- Command line arguments
opt = op:parse()
op:summarize()

transHeight = tonumber(opt.transHeight)
transWidth = tonumber(opt.transWidth)
noOfTrans = tonumber(opt.noOfTrans)
translate = opt.translate
patchHeight = tonumber(opt.patchHeight)
patchWidth = tonumber(opt.patchWidth)
noOfClutters = tonumber(opt.noOfClutters)
outdir = opt.outdir
os.execute("mkdir " .. outdir)
saveImage = opt.saveImage
if saveImage then
   trainDir = paths.concat(outdir, "train")
   os.execute("mkdir " .. trainDir)
   testDir = paths.concat(outdir, "test")
   os.execute("mkdir " .. testDir)
end

-- Load training data
ds = dp.Mnist{}
imagesTrain = ds:get('train','input', 'bchw','float')
labelsTrain = ds:get('train','target')
imagesValid = ds:get('valid','input', 'bchw','float')
labelsValid = ds:get('valid','target')

-- Training+Validation data
trainDictFile = paths.concat(outdir, "trainDict.t7")
images = torch.cat(imagesTrain, imagesValid, 1)
labels = torch.cat(labelsTrain, labelsValid, 1)
trainDict = {}
print("Translating/Cluttering digits")
local noOfSamples = images:size(1)
local channels = images:size(2)
local height = images:size(3)
local width = images:size(4)
local transImages = torch.zeros(noOfSamples*noOfTrans,
                                channels, transHeight, transWidth)
local transLabels = torch.zeros(noOfSamples*noOfTrans)
local indx = 1
local sample
for i=1, noOfSamples do
   for j=1,noOfTrans do
      if translate then
         sample = translate_digit(images[i], transHeight, transWidth)
      else
         sample = cluttered_digit(images[i], images, noOfClutters,
                                  transHeight, transWidth,
                                  patchHeight, patchWidth)
      end
      transImages[{{indx}}]:copy(sample)
      transLabels[indx] = labels[i]
      indx = indx + 1
   end
end
print(indx, noOfSamples*noOfTrans)
trainDict["images"] = transImages
trainDict["labels"] = transLabels
torch.save(trainDictFile, trainDict)

-- Save train images to disk
if saveImage then
   print("Saving train images to disk.")
   local noOfSamples = trainDict["images"]:size(1)
   local labelsDirs = {}
   local labelsDir
   for i=0, 9 do
      labelsDir = paths.concat(trainDir, tostring(i))
      os.execute("mkdir " .. labelsDir)
      labelsDirs[i+1] = labelsDir
   end
   local indx = 1
   for i=1, noOfSamples do
      image.save(paths.concat(labelsDirs[trainDict["labels"][i]],
                              tostring(indx)..".png"), trainDict["images"][i])
      indx = indx + 1
   end
end

-- Testing data
testDictFile = paths.concat(outdir, "testDict.t7")
images = ds:get('test','input', 'bchw','float')
labels = ds:get('test','target')
testDict = {}
collectgarbage()
print("Translating/Cluttering digits")
local noOfSamples = images:size(1)
local channels = images:size(2)
local height = images:size(3)
local width = images:size(4)
local transImages = torch.zeros(noOfSamples*noOfTrans,
                                channels, transHeight, transWidth)
local transLabels = torch.zeros(noOfSamples*noOfTrans)
local indx = 1
local sample
for i=1, noOfSamples do
   for j=1,noOfTrans do
      if translate then
         sample = translate_digit(images[i], transHeight, transWidth)
      else
         sample = cluttered_digit(images[i], images, noOfClutters,
                                  transHeight, transWidth,
                                  patchHeight, patchWidth)
      end
      transImages[{{indx}}]:copy(sample)
      transLabels[indx] = labels[i]
      indx = indx + 1
   end
end
print(indx, noOfSamples*noOfTrans)
testDict["images"] = transImages
testDict["labels"] = transLabels
torch.save(testDictFile, testDict)

-- Save test images to disk
if saveImage then
   print("Saving test images to disk.")
   local noOfSamples = testDict["images"]:size(1)
   local labelsDirs = {}
   local labelsDir
   for i=0, 9 do
      labelsDir = paths.concat(testDir, tostring(i))
      os.execute("mkdir " .. labelsDir)
      labelsDirs[i+1] = labelsDir
   end
   local indx = 1
   for i=1, noOfSamples do
      image.save(paths.concat(labelsDirs[testDict["labels"][i]],
                              tostring(indx)..".png"), testDict["images"][i])
      indx = indx + 1
   end
end
