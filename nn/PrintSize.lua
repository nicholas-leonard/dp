local PrintSize, parent = torch.class('nn.PrintSize', 'nn.Module')

function PrintSize:__init(prefix)
   parent.__init(self)
   self.prefix = prefix
end

function PrintSize:updateOutput(input)
   self.output = input
   print(self.prefix..":input\n", input:size())
   return self.output
end


function PrintSize:updateGradInput(input, gradOutput)
   print(self.prefix.."gradOuput\n", gradOutput:size())
   self.gradInput = gradOutput
   return self.gradInput
end
