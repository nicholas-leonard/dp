require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Script to harmonize downloaded ImageNet validation and training sets (ILSVRC2014 Classification)')
cmd:text('The training set class directories are renumbered by class index.')
cmd:text('The validation set images are moved into directories renamed by class index.')
cmd:text('And we create a file in each set mapping class indexes (a number between 1 and 1000) and the class name + description.')
cmd:text('Ref.: ILSVRC2012_devkit/readme.txt')
cmd:text('This script should be preceded by the downloadimagenet.lua script')
cmd:text('Example:')
cmd:text('$> th harmonizeimagenet.lua ')
cmd:text('Options:')
cmd:option('--dataPath', paths.concat(dp.DATA_DIR,'ImageNet'), 'data path')
cmd:option('--metaPath', '', 'path to meta directory (defaults to opt.dataPath/metadata)')
cmd:option('--trainPath', '', 'path to training images (defaults to opt.dataPath/ILSVRC2012_img_train)')
cmd:option('--validPath', '', 'path to validation images (defaults to opt.dataPath/ILSVRC2012_img_val)')
cmd:option('--devkitPath', '', 'path to the devkit, it contains the a blacklist and groundtruth labels (defaults to opt.dataPath/ILSVRC2014_devkit)')
cmd:option('--forReal', false, 'move everything for real')
cmd:option('--progress', false, 'display progress bars')
cmd:text()
opt = cmd:parse(arg or {})

opt.metaPath = opt.metaPath == '' and paths.concat(opt.dataPath,'metadata') or opt.metaPath
opt.trainPath = opt.trainPath == '' and paths.concat(opt.dataPath,'ILSVRC2012_img_train') or opt.trainPath
opt.validPath = opt.validPath == '' and paths.concat(opt.dataPath,'ILSVRC2012_img_val') or opt.validPath
opt.devkitPath = opt.devkitPath == '' and paths.concat(opt.dataPath,'ILSVRC2014_devkit') or opt.devkitPath

assert(paths.dirp(opt.metaPath), "metaPath doesn't exist ")
assert(paths.dirp(opt.trainPath), "trainPath doesn't exist ")
assert(paths.dirp(opt.validPath), "validPath doesn't exist ")
assert(paths.dirp(opt.devkitPath), "devkitPath doesn't exist ")


-- this contains data taken from the devkit/meta_clsloc.mat
--[[ 1x1860 struct array with fields:
       ILSVRC2014_ID (classIdx)
       WNID (synsetId)
       words
       gloss
       num_children
       children
       wordnet_height
       num_train_images
--]]

-- this file was downloaded via the downloadimagenet.lua script
metaPath = paths.concat(opt.metaPath, 'json.th7')
metadata = torch.load(metaPath)

synsetIndex = {}
classIndex = {}

print"reading synset metadata"
synsets = metadata.synsets[1]
for i, synset in ipairs(synsets) do
   local classIdx, synsetId, words, gloss, nChildren, children = unpack(synset)
   classIdx = classIdx[1][1]
   synsetId = synsetId[1]
   words = words[1]
   gloss = gloss[1]
   nChildren = nChildren[1][1]
   children = children[1]
   synsetIndex[synsetId] = classIdx
   classIndex[classIdx] = {synsetId, words, gloss, nChildren, children}
end

--[[
There are 50 validation images for each synset.
The classification ground truth of the validation images is in 
    data/ILSVRC2014_clsloc_validation_ground_truth.txt,
where each line contains one ILSVRC2014_ID (classIdx) for one image, in the
ascending alphabetical order of the image file names.
--]]
print"reading validation set"
validFiles = {}
for validFile in lfs.dir(opt.validPath) do
   if paths.filep(paths.concat(opt.validPath, validFile)) then
      table.insert(validFiles, validFile)
   end
end

print"sorting validation filenames"
--ascending alphabetical order of the image file names
_.sort(validFiles)

-- groundtruth for validation set
groundtruthPath = paths.concat(opt.devkitPath,"data/ILSVRC2014_clsloc_validation_ground_truth.txt")

i = 1

if #validFiles > 0 then
   print"moving validation files into class directories"
   for line in io.lines(groundtruthPath) do
      local classIdx = tonumber(line)
      assert(classIdx <= 1000 and classIdx > 0)
      local validFile = validFiles[i]
      local srcPath = paths.concat(opt.validPath, validFile)
      local dstPath = paths.concat(opt.validPath, 'class'..classIdx)
      local cmd = string.format("mv %s %s", srcPath, paths.concat(dstPath, validFile))
      if opt.forReal then
         paths.mkdir(dstPath)
         assert(paths.dirp(dstPath))
         os.execute(cmd)
      end
      if opt.progress then
         xlua.progress(i, #validFiles)
      end
      i = i + 1
   end
end

print"renaming training directories to 'class'..classIdx"
i = 1
for synsetDir in lfs.dir(opt.trainPath) do
   local srcPath = paths.concat(opt.trainPath, synsetDir)
   if paths.dirp(srcPath) and synsetIndex[synsetDir] then
      local dstPath = paths.concat(opt.trainPath, 'class'..synsetIndex[synsetDir])
      assert(not dp.is_file(dstPath), string.format("%s already exists", dstPath))
      cmd = string.format("mv %s %s", srcPath, dstPath)
      if opt.forReal then
         os.execute(cmd)
      end
   end
   if opt.progress then
      xlua.progress(i, 1000)
   end
   i = i + 1
end
 
print"saving classInfo.th7 metadata"
-- save a copy of the classIndex in the metaPath
if opt.forReal then
   torch.save(paths.concat(opt.metaPath, 'classInfo.th7'), classIndex)
end
