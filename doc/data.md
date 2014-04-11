# Data and preprocessing #
DataTensor, DataSet, DataSource, Samplers and Preprocessing.
DataSource is made up of 3 DataSet instances : train, valid and test.

<a name="dp.DataTensor"/>
## DataTensor ##
Encapsulates a torch.Tensor. Provides access to it using different
viewing methods. A view may reshape the tensor inplace and render it
contiguous. Views can be used to convert data into new axes formats 
using torch.Tensor:resize, :transpose, :contiguous. The 
conversions may be done in-place (default), or may be simply  
returned using the conversion methods (feature, class, image, etc.). 
A DataTensor may also holds metadata.

<a name="dp.DataTensor.__init"/>
### dp.DataTensor(data, [axes, sizes]) ###
Constructs a dp.DataTensor out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs :
```dt = dp.DataTensor{data=torch.Tensor(3,4), axes={'b','f'}, sizes={3,4}}```

`data` is a torch.Tensor with at least 1 dimensions. 

`axes` is a table defining the order and nature of each dimension of a tensor. Two common examples would be the archtypical MLP input : `{'b', 'f'}`, 
or a common image representation : `{'b', 'h', 'w', 'c'}`. 
Possible axis symbols are : 
 1. Standard Axes: 
  * `'b'` : Batch/Example 
  * `'f'` : Feature 
  * `'t'` : Class 
 2. Image Axes 
  * `'c'` : Color/Channel 
  * `'h'` : Height 
  * `'w'` : Width 
  * `'d'` : Dept 

The provided `axes` should be the most expanded version of the storage. For example, while an image can be represented as a vector, in which case it takes the form of `{'b','f'}`, its expanded axes format could be `{'b', 'h', 'w', 'c'}`. Defaults to `{'b','f'}`.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage. A table or torch.LongTensor holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. Defaults to data:size().

<a name="dp.DataTensor.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b', 'f'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b', 'f'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.

<a name="dp.DataSet"/>
## DataSet ##
TODO

<a name="dp.DataSource"/>
## DataSource ##
TODO


