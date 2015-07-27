------------------------------------------------------------------------
--[[ BillionWords ]]--
-- The corpus is derived from the 
-- training-monolingual.tokenized/news.20??.en.shuffled.tokenized data 
-- distributed at http://statmt.org/wmt11/translation-task.html.
-- We use the preprocessing suggested by 
-- https://code.google.com/p/1-billion-word-language-modeling-benchmark
------------------------------------------------------------------------
local BillionWords, parent = torch.class("dp.BillionWords", "dp.DataSource")
BillionWords.isBillionWords = true

BillionWords._name = 'BillionWords'
BillionWords._sentence_start = 793470 --"<S>"
BillionWords._sentence_end = 793471 --"</S>"
BillionWords._unknown_word = 793469 --"<UNK>"
BillionWords._root_id = 880542

function BillionWords:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, load_all
   args, self._context_size, self._train_file, self._valid_file, 
         self._test_file, self._word_file, self._data_path, 
         self._recurrent, self._download_url, load_all 
      = xlua.unpack(
      {config},
      'BillionWords', 
      'A dataset of one billion words used for language modeling',
      {arg='context_size', type='number', default=1,
       help='number of previous words to be used to predict the next one.'},
      {arg='train_file', type='string', default='train_data.th7',
       help='name of training file'},
      {arg='valid_file', type='string', default='valid_data.th7',
       help='name of validation file'},
      {arg='test_file', type='string', default='test_data.th7',
       help='name of test file'},
      {arg='word_file', type='string', default='word_map.th7',
       help='name of file containing mapping of word_ids to word strings.'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='recurrent', type='number', default=false,
       help='For RNN training, set this to true. In which case, '..
       'outputs a target word for each input word'},
      {arg='download_url', type='string',
       default='http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz',
       help='URL from which to download dataset if not found on disk.'},
      {arg='load_all', type='boolean', 
       help='Load all datasets : train, valid, test.', default=true}
   )
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadTest()
   end
   parent.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
   })
end

function BillionWords:loadTrain()
   local data = self:loadData(self._train_file, self._download_url)
   self:trainSet(self:createSentenceSet(data,'train'))
   return self:trainSet()
end

function BillionWords:loadValid()
   local data = self:loadData(self._valid_file, self._download_url)
   self:validSet(self:createSentenceSet(data,'valid'))
   return self:validSet()
end

function BillionWords:loadTest()
   local data = self:loadData(self._test_file, self._download_url)
   self:testSet(self:createSentenceSet(data,'test'))
   return self:testSet()
end

--Creates an BillionWords Dataset out of data and which_set
function BillionWords:createSentenceSet(data, which_set)
   -- construct dataset
   return dp.SentenceSet{
      data=data, which_set=which_set, context_size=self._context_size,
      end_id=self._sentence_end, start_id=self._sentence_start, 
      words=self._classes, recurrent=self._recurrent
   }
end

-- Get the raw data. The first column indicates the start of each context
function BillionWords:loadData(file_name, download_url)
   if not self._classes then
      local path = parent.getDataPath{
         name=self._name, url=download_url, 
         decompress_file=self._word_file, data_dir=self._data_path
      }
      self._classes = torch.load(path)
   end
   local path = parent.getDataPath{
      name=self._name, url=download_url, decompress_file=file_name, 
      data_dir=self._data_path
   }
   local tensor = torch.load(path)
   assert(tensor:size(2) == 2)
   collectgarbage()
   return tensor
end

function BillionWords:vocabulary()
   return self._classes
end

function BillionWords:vocabularySize()
   return table.length(self._classes)
end

-- this can be used to initialize a SoftMaxTree (optional)
function BillionWords:hierarchy(file_name)
   file_name = file_name or 'word_tree1.th7'
   local hierarchy = torch.load(paths.concat(self._data_path, self._name, file_name))
   _.map(hierarchy, function(k,v) 
      assert(torch.type(k) == 'number', 
         "Hierarchy keys should be numbers")
      assert(torch.type(v) == 'torch.IntTensor', 
         "Hierarchy values should be torch.IntTensors")
   end)
   return hierarchy
end

-- this can be used to initialize a SoftMaxTree
function BillionWords:frequencyTree(file_name, binSize)
   file_name = file_name or 'word_freq.th7'
   binSize = binSize or 100
   local wf = torch.load(paths.concat(self._data_path, self._name, file_name))
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

function BillionWords:rootId()
   return self._root_id
end
