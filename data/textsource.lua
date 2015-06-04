------------------------------------------------------------------------
--[[ TextSource ]]--
-- Creates a DataSource out of 3 strings or text files.
-- Text files are assumed to be arranged one sentence per line.
------------------------------------------------------------------------
local TextSource, parent = torch.class("dp.TextSource", "dp.DataSource")
TextSource.isTextSource = true

function TextSource:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args
   args, self._name, self._context_size, self._recurrent,
         self._train_file, self._valid_file, self._test_file, 
         self._data_path, self._download_url, self._string
      = xlua.unpack(
      {config},
      'TextSource', 
      'Creates a DataSource out of 3 strings or text files',
      {arg='name', type='string', req=true,
       help='name of datasource. Also name of directory in data_path'},
      {arg='context_size', type='number', default=1,
       help='number of previous words to be used to predict the next one.'},
      {arg='recurrent', type='number', default=false,
       help='For RNN training, set this to true. In which case, '..
       'outputs a target word for each input word'},
      {arg='train_file', type='string', default='train_data.txt',
       help='name of training file'},
      {arg='valid_file', type='string', default='valid_data.txt',
       help='name of validation file'},
      {arg='test_file', type='string', default='test_data.txt',
       help='name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to train, valid, test files'},
      {arg='download_url', type='string',
       default='https://github.com/wojzaremba/lstm',
       help='URL from which to download dataset if not found on disk.'},
      {arg='string', type='boolean', default=false,
       help='set this to true when the *file args are the text itself'}
   )
   -- everything is loaded in the same order 
   -- to keep the mapping of word to word_id consistent
   self:setTrainSet(self:textSetFromFile(self._train_file, 'train'))
   self:setValidSet(self:testSetFromFile(self._valid_file, 'valid'))
   self:setTestSet(self:textSetFromFile(self._test_file, 'test'))
   
   parent.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
   })
end

function TextSource:textSetFromFile(file_name, which_set) 
   local data
   if not self._string then
      local path = parent.getDataPath{
         name=self._name, url=self._download_url, 
         decompress_file=file_name, data_dir=self._data_path
      }

      data = file.read(path)
      data = stringx.replace(data, '\n', '<eos>')
      data = stringx.split(data)
      
      local stringx = require('pl.stringx')
      local file = require('pl.file')
   else
      data = file_name
   end
   
   local word_id = 0
   self._word_freq = self._word_freq or {}
   self.vocab = self.vocab or {}
   
   local x = torch.zeros(#data)
   count_freq = (which_set ~= 'test')
   for i = 1, #data do
      local word = data[i]
      if not selfvocab[word] then
         word_id = word_id + 1
         self.vocab[word] = word_id
         self._word_freq[word_id] = 0
      end
      if count_freq then 
         self._word_freq[word_id] = self._word_freq[word_id] + 1
      end
      x[i] = word_id
   end
   
   self._classes = {}
   for word, word_id in pairs(self.vocab) do
      self._classes[word_id] = word
   end
   
   return dp.TextSet{
      data=x, which_set=which_set, context_size=self._context_size,
      recurrent = self._recurrent, words=self._classes
   }
end

function TextSource:vocabulary()
   return self._classes
end

function TextSource:vocabularySize()
   return table.length(self._classes)
end
