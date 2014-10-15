require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Display filter maps from a convolution neural network')
cmd:text('Example:')
cmd:text('$> th showfilters.lua --xpFile nps:26693:1408205007:1.dat ')
cmd:text('Options:')
cmd:option('--xpDir', dp.SAVE_DIR, 'directory containing experiment')
cmd:option('--xpFile', '', 'name of file containing experiment')
cmd:option('--savePath', '', 'where to save images')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

cutorch.setDevice(opt.useDevice) 
assert(xpFile ~= '', "missing xpFile argument")

xp = torch.load(paths.concat(opt.xpDir, opt.xpFile))
if opt.savePath == '' then
   opt.savePath = xpFile:gsub('.dat', '')
end
dp.check_and_mkdir(opt.savePath)

local model = xp:model()
-- get convolution
local conv = model:get(1)
assert(conv.isConvolution2D, "Expecting Convolution2D model on first layer")
-- get previous input images
local input = conv.input
assert(input.isImageView, "Expecting first input View to be ImageView")
input = input:forward('bchw')
-- get previous output images
local output = conv.output
assert(output.isImageView, "Expecting first output View to be ImageView")
output = output:forward('bchw')
-- get weights
assert(torch.type(conv._conv) == 'nn.SpatialConvolutionMM', "Expecting nn.SpatialConvolutionMM")
local weight = conv._conv.weight:view(-1, unpack(conv._kernel_size))

local filters = image.toDisplayTensor{input=weight, padding=2, nRow=math.ceil(math.sqrt(weight:size(1)))}
image.save(paths.concat(opt.savePath, 'filters.png'))

local inputs = image.toDisplayTensor{input=input, padding=2, nRow=math.ceil(math.sqrt(input:size(1)))}
image.save(paths.concat(opt.savePath, 'inputs.png'), inputs)

for i=1,output:size(1) do
   local outputs = image.toDisplayTensor{input=output[i], padding=2, nRow=math.ceil(math.sqrt(output:size(2)))}
   image.save(paths.concat(opt.savePath, 'img'..i..'_in.png'), input[i])
   image.save(paths.concat(opt.savePath, 'img'..i..'_out.png'), outputs)
   collectgarbage()
end

for i=1,output:size(2) do
   local outputs = output:select(2,i):contiguous()
   local filters = image.toDisplayTensor{input=outputs[i], padding=2, nRow=math.ceil(math.sqrt(output:size(1)))}
   image.save(paths.concat(opt.savePath, 'filter'..i..'_out.png'), filters)
   collectgarbage()
end


