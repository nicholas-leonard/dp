------------------------------------------------------------------------
--[[ Linear ]]--
-- Adapts a nn.Linear to the dp.Model interface
------------------------------------------------------------------------
local Linear, parent = torch.class("dp.Linear", "dp.Module")

function Linear:__init(config)
   config = config or {}
   local args, input_size, output_size, typename = xlua.unpack(
      {config},
      'Linear', 
      'Adapts a nn.Linear to the dp.Model interface',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='typename', type='string', default='linear', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   config.typename = config.typename or typename
   config.module = nn.Linear(input_size, output_size)
   parent.__init(self, config)
end

function Linear:setup(config)
   config.data_view = 'feature'
   parent.setup(self, config)
end

function Linear:maxNorm(max_out_norm, max_in_norm)
   if not self._backwarded then return end
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = param.weight.param
   if max_out_norm then
      -- rows feed into output neurons 
      dp.constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      dp.constrain_norms(max_in_norm, 1, weight)
   end
end
