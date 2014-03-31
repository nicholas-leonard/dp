require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run unit tests')
cmd:text('Example:')
cmd:text('$> th runtest.lua --units')
cmd:text('Options:')
cmd:option('--units', '', 'comma separated list of unit test function names')
cmd:text()
opt = cmd:parse(arg or {})

local tests
if opt.units ~= '' then 
   tests = _.split(opt.units, ',')
end
dp.test(tests)
