<a name="dp.Preprocess"/>
## Preprocessing ##

### Preprocess ###
Abstract class.

An object that can preprocess a [BaseTensor](data.md#dp.BaseTensor)
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
ds = Mnist{input_preprocess=dp.Standardize()
```

<a name="dp.Standardize"/>
### Standardize ###
A Preprocess that subtracts the mean and divides by the standard deviation.

<a name="dp.Standardize.__init"/>
### dp.Standardize{[global_mean, global_std, std_eps]} ###
Constructor.

`global_mean` is a boolean with a default value of `false`. When true, subtracts the (scalar) mean over every element in the datset. Otherwise, subtract the mean from each column (feature) separately. Uses the `BaseTensor:feature()` view.

`global_std` is a boolean with a default value of `false`. When true, after centering, divides by the (scalar) '..
standard deviation of every element in the design matrix. Otherwise, divide by the column-wise (per-feature) standard deviation.

`std_eps` is a number with a default value of `1e-4`. It is a stabilization factor added to the standard deviations before dividing. This prevents standard deviations very close to zero from causing the feature values to blow up too much.
