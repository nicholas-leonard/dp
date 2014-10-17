-----------------------------------------------------------------------
--[[ ZCA ]]--
-- Performs Zero Component Analysis Whitening.
-- Commonly used for images, but can be applied to anything.
-- http://ufldl.stanford.edu/wiki/index.php/Whitening
-----------------------------------------------------------------------
local ZCA = torch.class("dp.ZCA", "dp.Preprocess")
ZCA.isZCA = true

function ZCA:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args
   args, self._n_component, self._n_drop_component, self._filter_bias,
      self._progess, self._compute_undo = xlua.unpack(
      {config},
      'ZCA', 
      'ZCA whitening constructor',
      {arg='n_component', type='number',
       help='number of most important eigen components to use for ZCA'},
      {arg='n_drop_component', type='number', 
       help='number of least important eigen components to drop.'},
      {arg='filter_bias', type='number', default=0.1,
       help='Filters are scaled by 1/sqrt(filter_bias + variance)'},
      {arg='progress', type='boolean', default=true, 
       help='display progress bar'},
      {arg='compute_undo', type='boolean', default=false,
       help='computes inverse ZCA (for method undo())'}
   )
end

function ZCA:fit(X)
   assert (X:dim() == 2)
   local n_samples = X:size()[1]
         
   -- center data
   self._mean = X:mean(1)
   X:add(torch.mul(self._mean, -1):expandAs(X))

   if self._progress then
      print'computing ZCA'
   end
   local matrix = torch.mm(X:t(), X) / X:size(1)
   matrix:add(torch.eye(matrix:size(1)):mul(self._filter_bias)) 
   -- returns a eigen components 
   local eig_val, eig_vec = torch.eig(matrix, 'V')
   -- sort in descending order of importance (eigen values)
   local eig_idx
   eig_val, eig_idx = torch.sort(eig_val:select(2,1),1,true)
   eig_vec = eig_vec:index(2, eig_idx)
   
   if self._progress then
      print'done computing eigen values and vectors'
   end
   assert(eig_val:min() > 0)
   if self._n_component then
     eig_val = eig_val:sub(1, self._n_component)
     eig_vec = eig_vec:narrow(2, 1, self._n_component)
   end
   if self._n_drop_component then
      eig_val = eig_val:sub(self._n_drop_component, -1)
      local size = eig_vec:size(2)-self._n_drop_component
      eig_vec = eig_vec:narrow(2, self._n_drop_component, size)
   end
   
   if self._compute_undo then
      self._inv_P = torch.mm(
         torch.cmul(eig_vec, torch.pow(eig_val, 0.5):resize(1, eig_val:size(1)):expandAs(eig_vec)),
         eig_vec:clone():t()
      )
   end
   
   self._P = torch.mm(
      torch.cmul(eig_vec, eig_val:pow(-0.5):resize(1, eig_val:size(1)):expandAs(eig_vec)),
      eig_vec:t()
   )
   
   self._mean:mul(-1)
   assert(not _.isNaN(self._P:sum()))
end

function ZCA:apply(dv, can_fit)
   assert(dv.isDataView, "Expecting DataView")
   local X = dv:forward('bf')
   local new_X
   if can_fit then
      self:fit(X)
      new_X = torch.mm(X, self._P)
   else
      new_X = torch.mm(torch.add(X, self._mean:expandAs(X)), self._P)
   end
   dv:replace('bf', new_X)
end

function ZCA:undo(dv)
   assert(dv.isDataView, "Expecting DataView")
   local new_X = dv:forward('bf')
   local X = new_X.new():resizeAs(new_X)
   X:mm(new_X, self._inv_P)
   X:add(-1, self._mean:expandAs(X))
   return X
end
