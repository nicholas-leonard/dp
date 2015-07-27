-- we keep these tests appart since they may require downloads and 
-- require lots of memory
local mytester 
local dptest = {}
local mediator = dp.Mediator()

function dptest.BillionWords()
   local ds = dp.BillionWords{train_file='train_tiny.th7', context_size=10}
   local hierarchy = ds:hierarchy()
   mytester:assert(torch.type(hierarchy) == 'table')
end

function dptest.PennTreeBank()
   local ds = dp.PennTreeBank{context_size=20}
   local freqTree = ds:frequencyTree()
   local train = ds:trainSet()
   local valid = ds:validSet()
   local test = ds:testSet()
   mytester:assert(torch.type(train) == 'dp.TextSet')
   mytester:assert(torch.type(valid) == 'dp.TextSet')
   mytester:assert(torch.type(test) == 'dp.TextSet')
   mytester:assert(table.length(ds.vocab) == 10000)
end

function dp.testDatasets(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

