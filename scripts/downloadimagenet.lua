require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Script to download and extract ImageNet dataset (ILSVRC2014 Classification)')
cmd:text('You will require 360GB of space to complete the download and extraction process.') 
cmd:text('You can download each tarball in a different partition if you dont have that much space in one partition.') 
cmd:text('Example:')
cmd:text('$> th downloadimagenet.lua ')
cmd:text('Options:')
cmd:option('--savePath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'where to download and extract the files')
cmd:option('--metaURL', 'https://stife076.files.wordpress.com/2015/02/metadata.zip', 'URL for file containing serialized JSON mapping word net ids to class index, name and description')
cmd:option('--trainURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_train.tar', 'URL of train images')
cmd:option('--validURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_val.tar', 'URL of validation images')
cmd:option('--testURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_test.tar', 'URL of test images')
cmd:option('--devkitURL', 'http://image-net.org/image/ilsvrc2014/ILSVRC2014_devkit.tgz', 'URL of devkit')
cmd:option('--what', 'all', 'what to download : all, train, valid, test, devkit, meta')
cmd:option('--squash', false, 'squash existing downloaded files.')
cmd:text()
opt = cmd:parse(arg or {})

if opt.what == 'all' then 
   opt.urls = {opt.trainURL, opt.validURL, opt.testURL, opt.devkitURL, opt.metaURL}
else
   opt.urls = {opt[opt.what..'URL']}
end

dp.mkdir(opt.savePath)
assert(paths.dirp(opt.savePath), 'error creating --savePath')

for i, url in ipairs(opt.urls) do
   local tarName = paths.basename(url)
   local tarPath = paths.concat(opt.savePath, tarName)
   
   -- download file
   if paths.filep(tarPath) and not opt.squash then
      print(string.format("skipping download as dir %s already exists. Use --squash to squash", tarPath))
   else
      if paths.filep(tarPath) then
         os.execute("rm "..tarPath)
      end
      dp.do_with_cwd(opt.savePath, function() dp.download_file(url) end)
   end
   
   -- extract file
   local extractPath = paths.concat(opt.savePath, tarName:match("([^.]*)%."))
   if paths.dirp(extractPath) and not opt.squash then
      print(string.format("skipping extraction as dir %s already exists. Use --squash to squash", extractPath))
   else
      if paths.dirp(tarPath) then
         paths.rmdir(tarPath)
      end
      print(string.format("extracting downloaded file : %s", tarPath))
      
      dp.do_with_cwd(opt.savePath,
         function()
            dp.decompress_file(tarPath, (not tarPath:find('devkit')) and extractPath or nil)
         end)
      assert(paths.dirp(extractPath), string.format("expecting tar %s to be extracted at %s", tarName, extractPath))
      -- for the training directory, which contains a tar for each class
      for subTar in lfs.dir(extractPath) do 
         local subDir = subTar:match("([^.]*)%.tar")
         if subDir then
            local subExtractPath = paths.concat(extractPath, subDir)
            local subTarPath = paths.concat(extractPath, subTar)
            if paths.dirp(subExtractPath) and not opt.squash then
               print(string.format("skipping extraction as dir %s already exists. Use --squash to squash", subExtractPath))
            else
               dp.decompress_tarball(subTarPath, subExtractPath)
            end
         end
      end
   end
end
