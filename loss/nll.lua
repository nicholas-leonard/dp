------------------------------------------------------------------------
--[[ NLL ]]--
-- Loss subclass
-- Adapter of nn.ClassNLLCriterion
-- Negative Log Likelihood 
------------------------------------------------------------------------
local NLL, parent = torch.class("dp.NLL", "dp.Loss")
NLL.isNLL = true

function NLL:__init(config)
   self._criterion = nn.ClassNLLCriterion()
   config = config or {}
   config.target_type = config.target_type or 'torch.IntTensor'
   config.target_view = 'b'
   config.input_view = 'bf'
   parent.__init(self, config)
end

function NLL:_type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' then
      self._input_type = type
      self._criterion:type(type)
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      self._output_type = type
   end
end
