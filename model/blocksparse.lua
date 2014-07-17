------------------------------------------------------------------------
--[[ BlockSparse ]]--
-- A n-layer model for Distributed Conditional Computation
------------------------------------------------------------------------
local BlockSparse, parent = torch.class("dp.BlockSparse", "dp.Layer")
BlockSparse.isBlockSparse = true

function BlockSparse:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, n_block, hidden_size, window_size, gater_size, 
      output_size, noise_std, gater_act, expert_act, gater_style, 
      expert_scale, gater_scale, interleave, threshold_lr, alpha_range, 
      sparse_init, typename = xlua.unpack(
      {config},
      'BlockSparse', 
      'Deep mixture of experts model. It is n parametrized '..
      'layers of gated experts. There are n-1 gaters, one for each '..
      'layer of hidden neurons between nn.BlockSparses.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='n_block', type='table', req=true,
       help='number blocks in hidden layers between nn.BlockSparses.'},
      {arg='hidden_size', type='table', req=true,
       help='number of neurons per block in hidden layers between nn.BlockSparses.'},
      {arg='window_size', type='table', req=true,
       help='number of blocks used per example in each layer.'},
      {arg='gater_size', type='table', req=true,
       help='number of neurons in gater hidden layers'},
      {arg='output_size', type='number', req=true,
       help='output size of last layer'},
      {arg='noise_std', type='number', req=true,
       help='std deviation of gaussian noise used for NoisyReLU'},
      {arg='gater_act', type='nn.Module', default='',
       help='gater hidden activation. Defaults to nn.Tanh()'},
      {arg='expert_act', type='table', default='',
       help='expert activation. Defaults. to nn.Tanh()'},
      {arg='gater_style', type='string', default='NoisyReLU',
       help='comma-separated sequence of Modules to use for gating'},
      {arg='expert_scale', type='number', default=1,
       help='scales the learningRate for the experts'},
      {arg='gater_scale', type='number', default=1,
       help='scales the learningRate for the gater'},
      {arg='interleave', type='boolean', default=false,
       help='when true, alternate between training gater and experts every epoch'},
      {arg='threshold_lr', type='number', default=0.1,
       help='learning rate to get the optimum threshold for a desired sparsity'},
      {arg='alpha_range', type='table', default='',
       help='{start_alpha, num_batches, endsall}'},
      {arg='sparse_init', type='boolean', default=false,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='typename', type='string', default='BlockSparse', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._gater_size = gater_size
   self._n_block = n_block
   self._hidden_size = hidden_size
   self._window_size = window_size
   self._interleave = interleave
   self._expert_scale = expert_scale
   self._gater_scale = gater_scale
   self._expert_phase = true
   alpha_range = (alpha_range == '') and {0.5, 1000, 0.01} or alpha_range
   gater_act = (gater_act == '') and nn.Tanh() or gater_act
   expert_act = (expert_act == '') and nn.Tanh() or expert_act
   
   -- experts
   self._experts = {}
   local para = nn.ParallelTable()
   para:add(expert_act:clone())
   para:add(nn.Identity())
   
   local expert = nn.Sequential()
   expert:add(nn.BlockSparse(1, self._input_size, self._n_block[1], self._hidden_size[1], config.acc_update))
   expert:add(para)
   table.insert(self._experts, expert)
   
   for i=1,#self._n_block - 1 do
      expert = nn.Sequential()
      expert:add(nn.BlockSparse(self._n_block[i], self._hidden_size[i], self._n_block[i+1], self._hidden_size[i+1], config.acc_update))
      expert:add(para:clone())
      table.insert(self._experts, expert)
   end
   
   local dept = #self._n_block
   expert = nn.Sequential()
   expert:add(nn.BlockSparse(self._n_block[dept], self._hidden_size[dept], 1, output_size, config.acc_update))
   expert:add(expert_act:clone())
   table.insert(self._experts, expert)
   
   -- gaters
   self._gates = {}
   
   self._gater = nn.Sequential()
   local inputSize = self._input_size
   for i, gaterSize in ipairs(self._gater_size) do
      self._gater:add(nn.Linear(inputSize, gaterSize))
      self._gater:add(gater_act:clone())
      inputSize = gaterSize
   end
   
   local concat = nn.ConcatTable()
   self._relus = {}
   self._balances = {}
   for i=1,#self._window_size do
      local gate = nn.Sequential()
      gate:add(nn.Linear(inputSize, self._n_block[i]))
      for i,gater_str in ipairs(_.split(gater_style, '[,]')) do
         if gater_str == 'NoisyReLU' then
            local relu = nn.NoisyReLU(self._window_size[i]/self._n_block[i], threshold_lr, alpha_range, noise_std[i])
            gate:add(relu)
            table.insert(self._relus, relu)
         elseif gater_str == 'Balance' then
            local balance = nn.Balance(10)
            gate:add(balance)
            table.insert(self._balances, balance)
         elseif gater_str == 'SoftMax' then
            gater:add(nn.SoftMax())
         else
            error("unknown gater_style:"..gater_str, 2)
         end
      end
      gate:add(nn.LazyKBest(self._window_size[i]))
      concat:add(gate)
      table.insert(self._gates, gate)
   end
   self._gater:add(concat)
   
   -- mixture
   gater_scale = interleave and 0 or gater_scale
   self._bm = nn.BlockMixture(self._experts, self._gater, expert_scale, gater_scale)
   self._module = self._bm
   
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'bf'
   config.tags = config.tags or {}
   config.sparse_init = sparse_init
   parent.__init(self, config)
   -- only works with cuda
   self:type('torch.CudaTensor')
end

function BlockSparse:doneEpoch(report, ...)
   self:zeroStatistics()
   if self._interleave then
      self._expert_phase = not self._expert_phase
      self._bm.expertScale = self._expert_phase and self._expert_scale or 0
      self._bm.gaterScale = (not self._expert_phase) and self._gater_scale or 0
   end
end

function BlockSparse:report()
   if not self._next_report then
      self._next_report = true
      return
   end
   if self._expert_phase then
      print"Expert Phase"
   else
      print"Gater Phase"
   end
   local nVal = 5
   for i, relu in ipairs(self._relus) do
      local vals = relu.sparsity:select(1,1):float():sort()
      print(i, 
         table.tostring(vals:narrow(1,1,nVal):clone():storage():totable()), 
         table.tostring(vals:narrow(1,vals:size(1)-nVal+1,nVal):clone():storage():totable())
      )
   end
end
