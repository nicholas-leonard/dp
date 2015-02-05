require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Script to download and extract ImageNet dataset (ILSVRC2014 Classification)')
cmd:text('Example:')
cmd:text('$> th downloadImageNet.lua ')
cmd:text('Options:')
cmd:option('--savePath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'where to download and extract the files')
cmd:option('--trainURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_train.tar', 'URL of train images')
cmd:option('--validURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_val.tar', 'URL of validation images')
cmd:option('--testURL', 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_test.tar', 'URL of test images')
cmd:option('--devkitURL, ''http://image-net.org/image/ilsvrc2014/ILSVRC2014_devkit.tgz', 'URL of devkit')
cmd:option('--what', 'all', 'what to download : all, train, valid, test, devkit')
cmd:option('--squash' false, 'squash existing downloaded files.')
cmd:text()
opt = cmd:parse(arg or {})

if opt.what == 'all' then 
   opt.urls = {opt.trainURL, opt.validURL, opt.testURL, opt.devkitURL}
else
   opt.urls = {opt[opt.what..'URL']}
end
   
paths.mkdir(opt.savePath)

for i, url in ipairs(opt.urls) do
   local tarName = paths.basename(url)
   local tarPath = paths.concat(opt.savePath, tarName)
   if paths.filep(tarPath) and not opt.squash then
      print(string.format("skipping %s as it already exists. Use --squash to squash", url))
   else
      dp.do_with_cwd(
         paths.dirname(opt.savePath), 
         function() dp.download_file(url) end
      )
      print(string.format("extracting downloaded tar file : %s", tarPath))
      dp.decompress_tarball(tarPath)
   end
end
