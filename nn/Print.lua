local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(prefix)
   parent.__init(self)
   self.prefix = prefix
end

function Print:updateOutput(input)
   self.output = input
   print(self.prefix..":input\n", input.size and input:size() or input, torch.type(input))
   return self.output
end


function Print:updateGradInput(input, gradOutput)
   print(self.prefix.."gradOutput\n", gradOutput.size and gradOutput:size() or gradOutput, torch.type(gradOutput))
   self.gradInput = gradOutput
   return self.gradInput
end
