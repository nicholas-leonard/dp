------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a transfer function.
-- For a linear transformation, you can use nn.Identity.
------------------------------------------------------------------------
local Neural, parent = torch.class("dp.Neural", "dp.Layer")
Neural.isNeural = true

function Neural:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, transfer, typename 
      = xlua.unpack(
      {config},
      'Neural', 
      'An affine transformation followed by a transfer function.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._transfer = transfer
   self._affine = nn.Linear(input_size, output_size)
   self._module = nn.Sequential()
   self._module:add(self._affine)
   self._module:add(self._transfer)
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = config.input_view or 'bf'
   config.output_view = config.output_view or 'bf'
   parent.__init(self, config)
end

