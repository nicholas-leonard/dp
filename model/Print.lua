local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(prefix)
   parent.__init(self)
   self.prefix = prefix
end

function Print:updateOutput(input)
   self.output = input
   print(self.prefix..":input", input)
   return self.output
end


function Print:updateGradInput(input, gradOutput)
   print(self.prefix.."gradOuput", gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
