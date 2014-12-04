require 'dp'
ds = dp.FacialKeypoints()
data = ds:loadData('train.th7', ds._download_url)
targets = data:narrow(2, 1, 30)
sum = torch.Tensor(30):zero()
count = torch.Tensor(30):zero()
for k=1,targets:size(1) do
   local target = targets[k]
   for i=1,targets:size(2) do
      local kp = target[i]
      if kp ~= -1 then
         sum[i] = sum[i] + kp
         count[i] = count[i] + 1
      end
   end
end
baseline = torch.cdiv(sum, count)
print(baseline)
torch.save('baseline.th7', baseline)

kaggle = dp.FKDKaggle{
   submission = ds:loadSubmission(), 
   file_name = 'baseline.csv'
}


testSet = ds:testSet()
nTest = testSet:nSample()
bsView = baseline:view(1,30)
blurView = ds:makeTargets(bsView)
output = dp.SequenceView('bwc', blurView:expand(nTest, 30, 98))
batch = testSet:sub(1,nTest)
kaggle:add(batch, output, {}, {})
kaggle:errorMinima(true)

