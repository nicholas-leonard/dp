local AddConstant, parent = torch.class('nn.AddConstant', 'nn.Module')

function AddConstant:__init(scalar)
   parent.__init(self)
   self.scalar = scalar
end

function AddConstant:updateOutput(input)
   self.gradInput:resize(input:size())
   self.output:resize(input:size()) 
   
   self.output:copy(input);
   self.output:add(self.scalar)
   return self.output
end 

function AddConstant:updateGradInput(input, gradOutput)
   self.gradInput:copy(gradOutput)
   return self.gradInput
end
