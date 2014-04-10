-- updates parameters inverse-proportionally to 
-- number of times each index was used since last update.
-- less forward/backwards --> higher learning rate (because these are 
-- downscaled proportionally to batch size using scale, in criterion, 
-- or learning rate))
local FairLookupTable, parent = torch.class('nn.FairLookupTable', 'nn.LookupTable')

function FairLookupTable:__init(batchScaled, nIndex, ...)
   parent.__init(self, nIndex, ...)
   
   batchSize = torch.LongTensor(#self.size + 1)
   batchSize:narrow(1, 2,#self.size):copy(torch.LongTensor(self.size))
   batchSize[1] = 1
   self.batchSize = batchSize:storage()
   
   -- when this is true, assumes that learningRate, scale or criterion
   -- already scales the resulting update doing the equivalent of 
   -- dividing it by the number of examples in the batch.
   self.batchScaled = batchScaled
end

function FairLookupTable:scaleUpdateByKey(inputKey)
   local nBackward = self.inputs[inputKey]
   local kscale 
   if self.batchScaled then 
      kscale = self.nBackward/nBackward
   else
      kscale = 1/nBackward
   end
   return kscale
end
