------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a non-linearity
------------------------------------------------------------------------
local Neural, parent = torch.class("dp.Neural", "dp.Model")
Neural.isNeural = true

function Neural:__init(config)
   config = config or {}
   local args, input_size, output_size, transfer, dropout, typename 
      = xlua.unpack(
      {config},
      'Neural', 
      'An affine transformation followed by a non-linearity',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._transfer = transfer
   config.typename = config.typename or typename
   self._affine = nn.Linear(input_size, output_size)
   self._dropout = dropout
   parent.__init(self, config)
   self._tags.hasParams = true
   self._uncuda = (torch.typename(self._transfer) == 'nn.SoftMax')
   self:zeroGradParameters()
   self:checkParams()
end

function Neural:checkParams()
   for param_name,param_table in pairs(self:parameters()) do
      for k,v in pairs(param_table) do
         assert(not _.isNaN(v:sum()), 
                "NaN Error for " .. k .. " " .. param_name)
      end
   end
end

function Neural:setup(config)
   config.data_view = 'feature'
   parent.setup(self, config)
end

function Neural:_forward(gstate, eval)
   local activation = self.istate.act
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not eval)
      activation = self._dropout:forward(activation)
      self.mvstate.dropoutAct = activation
   end
   activation = self._affine:forward(activation)
   if self._uncuda then
      if self._recuda == nil then
         self._recuda = (activation:type() == 'torch.CudaTensor')
      end
      activation = activation:double()
   end
   self.mvstate.affineAct = activation
   self.ostate.act = self._transfer:forward(activation)
end

function Neural:_evaluate(gstate)
   -- requires for dropout
   self:_forward(gstate, true)
end

function Neural:_backward(gstate, scale)
   local input_act = self.mvstate.affineAct
   local output_grad = self.ostate.grad
   output_grad = self._transfer:backward(input_act, output_grad, scale)
   if self._recuda then
      output_grad = output_grad:cuda()
   end
   self.mvstate.affineGrad = output_grad
   input_act = self.mvstate.dropoutAct or self.istate.act
   output_grad = self._affine:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self.istate.act
      output_grad = self._dropout:backward(input_act,output_grad,scale)
   end
   self.istate.grad = output_grad
end

function Neural:_update(gstate)
   self._affine:updateParameters(gstate.learning_rate)
end

function Neural:zeroGradParameters()
   self._affine:zeroGradParameters()
end

function Neural:type(type)
   self._affine:type(type)
   if not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
   return parent.type(self, type)
end

function Neural:reset()
   return self._affine:reset()
end

-- TODO move this to __init() and test it!
function Neural:parameters()
   local params = self._params
   local module = self._affine
   if module.weight and module.weight:dim() ~= 0 then
      if not params.weight then
         params.weight = {}
      end
      params.weight.param=module.weight
      params.weight.grad=module.gradWeight
   end
   if module.bias and module.bias:dim() ~= 0 then
      if not params.bias then
         params.bias = {}
      end
      params.bias.param=module.bias
      params.bias.grad=module.gradBias
   end
   return params
end


function Neural:maxNorm(max_out_norm, max_in_norm)
   if not self.backwarded then return end
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = params.weight.param
   if max_out_norm then
      -- rows feed into output neurons 
      dp.constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      dp.constrain_norms(max_in_norm, 1, weight)
   end
end

function Neural:share(neural, ...)
   assert(neural.isNeural)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self._affine[v] ~= nil then
         self._affine[v]:set(neural:parameters()[v])
      end
   end
   return self      
end
