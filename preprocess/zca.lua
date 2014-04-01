-----------------------------------------------------------------------
--[[ ZCA ]]--
-- Performs Zero Component Analysis Whitening.
-- Used for images.
-- http://ufldl.stanford.edu/wiki/index.php/Whitening
-----------------------------------------------------------------------
local ZCA = torch.class("dp.ZCA", "dp.Preprocess")
ZCA.isZCA = true

function ZCA:__init(...)
   local args
   args, self.n_components, self.n_drop_components, self._filter_bias
      = xlua.unpack(
      {... or {}},
      'ZCA', 'ZCA whitening constructor',
      {arg='n_component', type='number',
       help='number of most important eigen components to use for ZCA'},
      {arg='n_drop_component', type='number', 
       help='number of least important eigen components to drop.'},
      {arg='filter_bias', type='number', default=0.1}
   )
end

function ZCA:fit(X)
   assert (X:dim() == 2)
   local n_samples = X:size()[1]
         
   -- center data
   self._mean = X:mean(1)
   X:add(-self._mean:expandAs(X))

   print'computing ZCA'
   local matrix = torch.mm(X:t(), X) / X:size(1)
   matrix:add(torch.eye(matrix:size(1)):mul(self._filter_bias)) 
   -- returns a eigen components in ascending order of importance
   local eig_val, eig_vec = torch.eig(matrix, 'V')
   local eig_val = eig_val:select(2,1)
   print'done computing eigen values and vectors'
   assert(eig_val:min() > 0)
   if self.n_components then
     eig_val = eig_val:sub(1, self.n_components)
     eig_vec = eig_vec:narrow(2, 1, self.n_components)
   end
   if self.n_drop_components then
      eig_val = eig_val:sub(self.n_drop_component, -1)
      local size = eig_vec:size(2)-self.n_drop_component
      eig_vec = eig_vec:narrow(2, self.n_drop_component, size)
   end
   self._P = torch.mm(
      torch.cmul(eig_vec, eig_val:pow(-0.5):reshape(1, eig_val:size(1)):expandAs(eig_vec)), 
      eig_vec:t()
   )
   assert(not _.isNaN(self._P:sum()))
end

function ZCA:apply(datatensor, can_fit)
   local X = datatensor:feature()
   local new_X
   if can_fit then
      self:fit(X)
      new_X = torch.mm(X, self._P)
   else
      new_X = torch.mm(torch.add(X, -self._mean:expandAs(X)), self._P)
   end
   datatensor:setData(new_X)
end
