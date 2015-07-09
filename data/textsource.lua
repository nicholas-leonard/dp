------------------------------------------------------------------------
--[[ TextSource ]]--
-- Creates a DataSource out of 1 to 3 strings or text files.
-- Text files are assumed to be arranged one sentence per line, each
-- line beginning with a space and ending with a space and a newline.

-- Feel free to send a Pull Request to extend this DataSource with 
-- your own constructor arguments and functionality.
------------------------------------------------------------------------
local TextSource, parent = torch.class("dp.TextSource", "dp.DataSource")
TextSource.isTextSource = true

function TextSource:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, train, valid, test
   args, self._name, self._context_size, self._recurrent, 
         self._bidirectional, train, valid, test, 
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
      {arg='bidirectional', type='boolean', default=false,
       help='For BiDirectionalLM, i.e. Bidirectional RNN/LSTMs, '..
       'set this to true. In which case, target = input'},
      {arg='train', type='string', 
       help='training text data or name of training file'},
      {arg='valid', type='string',
       help='validation text data or name of validation file'},
      {arg='test', type='string',
       help='test text data or name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to train, valid, test files'},
      {arg='download_url', type='string',
       help='URL from which to download dataset if not found on disk.'},
      {arg='string', type='boolean', default=false,
       help='set this to true when the *file args are the text itself'}
   )
   -- everything is loaded in the same order 
   -- to keep the mapping of word to word_id consistent
   self:trainSet(self:createTextSet(train, 'train'))
   self:validSet(self:createTextSet(valid, 'valid'))
   self:testSet(self:createTextSet(test, 'test'))
   
   parent.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
   })
end

function TextSource:createTextSet(file_name, which_set) 
   if file_name == nil or file_name == '' then
      return
   end
   local data
   if not self._string then
      if #file_name > 1000 then
         print("TextSource Warning : either you have a really long file_name "..
            "or you forget to add string=true to the constructor")
      end
      
      local path 
      if download_url then 
         path = parent.getDataPath{
            name=self._name, url=self._download_url, 
            decompress_file=file_name, data_dir=self._data_path
         }
      else
         path = paths.concat(self._data_path, file_name)
      end
      
      local file = require('pl.file')
      data = file.read(path)
   else
      data = file_name
   end
   
   local stringx = require('pl.stringx')
   data = stringx.replace(data, '\n', '<eos>')
   data = stringx.split(data)
   
   local word_seq = 0
   self._word_freq = self._word_freq or {}
   self.vocab = self.vocab or {}
   
   local x = torch.IntTensor(#data):fill(0)
   count_freq = (which_set ~= 'test')
   for i = 1, #data do
      local word = data[i]
      local word_id = self.vocab[word]
      if not word_id then
         word_seq = word_seq + 1
         self.vocab[word] = word_seq
         self._word_freq[word_seq] = 0
         word_id = word_seq
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
   
   -- Note that the context_size of a test set should probably be 
   -- manually set to 1 for recurrent models. This will allow 
   -- for feeding the test set as a single contiguous sequence of 
   -- words to your recurrent model (like RNN and LSTM).
   return dp.TextSet{
      data=x, which_set=which_set, context_size=self._context_size,
      recurrent=self._recurrent, bidirectional=self._bidirectional, 
      words=self._classes
   }
end

function TextSource:vocabulary()
   return self._classes
end

function TextSource:vocabularySize()
   return table.length(self._classes)
end

function TextSource:wordFrequency()
   return self._word_freq
end
