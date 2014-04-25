<a name="dp.Preprocess"/>
# Preprocessing #
Preprocessing is used to analyse and optimize the statistical properties of datasets such that they are easier to push through neural networks.

## Preprocess ##
Abstract class.

An object that can preprocess a [BaseTensor](data.md#dp.BaseTensor).
Preprocessing a basetensor implies changing the data that
a dataset actually stores. This can be useful to save
memory. If you know you are always going to access only
the same processed version of the dataset, it is better
to process it once and discard the original.

Preprocesses are capable of modifying many aspects of
a dataset. For example, they can change the way that it
converts between different formats of data. They can
change the number of examples that a dataset stores.
In other words, preprocesses can do a lot more than
just example-wise transformations of the examples stored
in a basetensor.

<a name="dp.Preprocess.apply"/>
### apply(basetensor, can_fit) ###
Abstract method.

`basetensor` is the BaseTensor to act upon.

`can_fit`. When true, the Preprocess can adapt internal parameters based on the contents of a basetensor.
This is usually true for basetensors taken from the training set. 

For example, let us preprocess the [Mnist](data.md#dp.Mnist) inputs. First, we load the datasource and create a [Standardize](#dp.Standardize) preprocess.
```lua
ds = dp.Mnist()
st = dp.Standardize()
```
Get the `train`, `valid` and `test` set inputs.
```lua
train = ds:trainSet():inputs()
valid = ds:validSet():inputs()
test = ds:testSet():inputs()
```
Fit and apply the preprocess to the `train` basetensor.
```lua
st:apply(train, true)
```
At this point the `st` preprocess has measured and stored some statistics on the `train` basetensor. Furthermore, the `train` basetensor has been preprocessed. We can apply the same preprocessing (with the same statistics) on the the `valid` and `test` basetensors.
```lua
st:apply(valid, false)
st:apply(test, false)
```
Since this is a common pattern in machine learning, we have simplified all this to one line of code.
```lua
ds = Mnist{input_preprocess=dp.Standardize()}
```

<a name="dp.Standardize"/>
## Standardize ##
A Preprocess that subtracts the mean and divides by the standard deviation.

<a name="dp.Standardize.__init"/>
### dp.Standardize{[global_mean, global_std, std_eps]} ###
Constructor.

`global_mean` is a boolean with a default value of `false`. When true, subtracts the (scalar) mean over every element in the datset. Otherwise, subtract the mean from each column (feature) separately. Uses the `BaseTensor:feature()` view.

`global_std` is a boolean with a default value of `false`. When true, after centering, divides by the (scalar) '..
standard deviation of every element in the design matrix. Otherwise, divide by the column-wise (per-feature) standard deviation.

`std_eps` is a number with a default value of `1e-4`. It is a stabilization factor added to the standard deviations before dividing. This prevents standard deviations very close to zero from causing the feature values to blow up too much.

<a name="dp.GCN"/>
## GCN ##
Performs Global Contrast Normalization by (optionally) subtracting the 
mean across features and normalizing by either 
the vector norm or the standard deviation (across features, for 
each example).

<a name="dp.GCN.__init"/>
### dp.GCN{[substract_mean, scale, sqrt_bias, use_std, min_divisor, batch_size]} ###
Construstor.

`substract_mean` is a boolean with a default value of true. 
Remove the mean across features/pixels before normalizing. 
Note that this is the per-example mean across pixels, not the per-pixel mean across examples.

`scale` is a number with a default value of 1.0. Multiply features by this constant.

`sqrt_bias` is a number with a default value of 0. A fudge factor added inside the square root.
Adds this amount inside the square root when computing the standard deviation or the norm.

`use_std` is a boolean with a default value of false. If True uses the standard deviation instead of the norm.

`min_divisor` is a number with a default value of 1e-8. If the divisor for an example is less than this value, do not apply it.

`batch_size` is a number with a default value 0. The size of a batch used internally.
       
Note that `sqrt_bias = 10`, `use_std = true` and defaults for all other
parameters corresponds to the preprocessing used in :
A. Coates, H. Lee and A. Ng. [An Analysis of Single-Layer
Networks in Unsupervised Feature Learning](http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf). AISTATS 14, 2011.
