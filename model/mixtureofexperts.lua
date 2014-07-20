------------------------------------------------------------------------
--[[ MixtureOfExperts ]]--
-- A n-layer model for Distributed Conditional Computation
------------------------------------------------------------------------
local MixtureOfExperts, parent = torch.class("dp.MixtureOfExperts", "dp.Layer")
MixtureOfExperts.isMixtureOfExperts = true

function MixtureOfExperts:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, n_expert, expert_size, expert_act, 
      gater_size, gater_act, output_size, output_act, sparse_init, 
      typename = xlua.unpack(
      {config},
      'MixtureOfExperts', 
      'Mixture of experts model.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='n_expert', type='number', req=true,
       help='Number of experts'},
      {arg='expert_size', type='table', req=true,
       help='number of neurons per expert hidden layer.'},
      {arg='expert_act', type='nn.Module', default='',
       help='expert activation. Defaults to nn.Tanh()'},
      {arg='gater_size', type='table', req=true,
       help='number of neurons in gater hidden layers'},
      {arg='gater_act', type='nn.Module', default='',
       help='gater hidden activation. Defaults to nn.Tanh()'},
      {arg='output_size', type='number', req=true,
       help='output size of last layer'},
      {arg='output_act', type='nn.Module', default='',
       help='output activation. Defaults to nn.LogSoftMax()'},
      {arg='sparse_init', type='boolean', default=false,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='typename', type='string', default='MixtureOfExperts', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._n_expert = n_expert
   self._expert_size = expert_size
   expert_act = (expert_act == '') and nn.Tanh() or expert_act
   self._gater_size = gater_size
   gater_act = (gater_act == '') and nn.Tanh() or gater_act
   self._output_size = output_size
   output_act = (output_act == '') and nn.LogSoftMax() or output_act
   
   -- experts
   self._experts = nn.ConcatTable()
   
   for i=1,self._n_expert do
      local inputSize = self._input_size
      local expert = nn.Sequential()
      for i,hiddenSize in ipairs(self._expert_size) do
         expert:add(nn.Linear(inputSize, hiddenSize))
         expert:add(expert_act:clone())
         inputSize = hiddenSize
      end
      expert:add(nn.Linear(inputSize, self._output_size))
      expert:add(output_act:clone())
      self._experts:add(expert)
   end
   
   -- gaters
   self._gater = nn.Sequential()
   local inputSize = self._input_size
   for i,hiddenSize in ipairs(self._gater_size) do
      self._gater:add(nn.Linear(inputSize, hiddenSize))
      self._gater:add(gater_act:clone())
      inputSize = hiddenSize
   end
   self._gater:add(nn.Linear(inputSize, self._n_expert))
   self._gater:add(nn.SoftMax())
   
   -- mixture
   self._trunk = nn.ConcatTable()
   self._trunk:add(self._gater)
   self._trunk:add(self._experts)
   self._mixture = nn.MixtureTable()
   self._module = nn.Sequential()
   self._module:add(self._trunk)
   self._module:add(self._mixture)
   
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'bf'
   config.tags = config.tags or {}
   config.sparse_init = sparse_init
   parent.__init(self, config)
end

