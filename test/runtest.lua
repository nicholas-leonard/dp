require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run unit tests')
cmd:text('Example:')
cmd:text('$> th test/runtest.lua --units "view,imageview"')
cmd:text('Options:')
cmd:option('--units', '', 'comma separated list of unit test function names')
cmd:option('--test', 'test', 'name of test function to run')
cmd:text()
opt = cmd:parse(arg or {})

local tests
if opt.units ~= '' then 
   tests = _.split(opt.units, ',')
end

dp[opt.test](tests)
