------------------------------------------------------------------------
--[[ PennTreeBank ]]--
-- The corpus is provided via https://github.com/wojzaremba/lstm
------------------------------------------------------------------------
local PennTreeBank, parent = torch.class("dp.PennTreeBank", "dp.TextSource")
PennTreeBank.isPennTreeBank = true

PennTreeBank._name = 'PennTreeBank'

function PennTreeBank:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, train_file, valid_file, test_file, url 
      = xlua.unpack(
      {config},
      'PennTreeBank', 
      'The Penn Tree Bank dataset',
      {arg='train_file', type='string', default='ptb.train.txt',
       help='name of training file'},
      {arg='valid_file', type='string', default='ptb.valid.txt',
       help='name of validation file'},
      {arg='test_file', type='string', default='ptb.test.txt',
       help='name of test file'},
      {arg='download_url', type='string',
       default='https://raw.githubusercontent.com/wojzaremba/lstm/master/data/',
       help='URL from which to download dataset if not found on disk.'}
   )
   config.name = config.name or self._name
   parent.__init(self, config)
   
   -- look for files locally
   local dsdir = paths.concat(self._data_path, self._name)
   dp.mkdir(dsdir)
   local train = paths.concat(dsdir, train_file)
   local valid = paths.concat(dsdir, valid_file)
   local test = paths.concat(dsdir, test_file)
   -- download if not found
   dp.check_and_download_file(train, url..train_file)
   dp.check_and_download_file(valid, url..valid_file)
   dp.check_and_download_file(test, url..test_file)
   -- initialize (order is important for consistency)
   self:trainSet(self:createTextSet(train, 'train'))
   self:validSet(self:createTextSet(valid, 'valid'))
   self:testSet(self:createTextSet(test, 'test'))
end

-- this can be used to initialize a SoftMaxTree
function PennTreeBank:frequencyTree(binSize)
   binSize = binSize or 100
   local wf = torch.IntTensor(self:wordFrequency())
   local vals, indices = wf:sort()
   local tree = {}
   local id = indices:size(1)
   function recursiveTree(indices)
      if indices:size(1) < binSize then
         id = id + 1
         tree[id] = indices
         return
      end
      local parents = {}
      for start=1,indices:size(1),binSize do
         local stop = math.min(indices:size(1), start+binSize-1)
         local bin = indices:narrow(1, start, stop-start+1)
         assert(bin:size(1) <= binSize)
         id = id + 1
         table.insert(parents, id)
         tree[id] = bin
      end
      recursiveTree(indices.new(parents))
   end
   recursiveTree(indices)
   return tree, id
end
