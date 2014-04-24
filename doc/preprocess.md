<a name="dp.Preprocess"/>
## Preprocess ##

### Preprocess ###

### Standardize ###
A Preprocess that subtracts the mean and divides by the standard deviation.

### Standardize{[global_mean, global_std, std_eps]} ###
`global_mean` is a boolean with a default value of `false`. When true, subtracts the (scalar) mean over every element in the datset. Otherwise, subtract the mean from each column (feature) separately. Uses the `BaseTensor:feature()` view.

`global_std` is a boolean with a default value of `false`. When true, after centering, divides by the (scalar) '..
standard deviation of every element in the design matrix. Otherwise, divide by the column-wise (per-feature) standard deviation.

`std_eps` is a number with a default value of `1e-4`. It is a stabilization factor added to the standard deviations before dividing. This prevents standard deviations very close to zero from causing the feature values to blow up too much.
