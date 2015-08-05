# Propagators #
These propagate Batches sampled from a DataSet using a Sampler through a 
[Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) 
in order to evaluate a Loss, provide Feedback or train the model :

  * [Propagator](#dp.Propagator) : abstract class;
    * [Optimizer](#dp.Optimizer) : optimizes a [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) on a `train` [DataSet](data.md#dp.DataSet);
    * [Evaluator](#dp.Evaluator) : evaluates a Model on a `valid` or `test` DataSet.

<a name="dp.Propagator"/>
[]()
## Propagator ##

Abstract Class for propagating a sampling distribution (a Sampler) through a 
[Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module). 
A Propagator can be sub-classed to build task-tailored training or evaluation algorithms.

<a name="dp.Propagator.__init"/>
[]()
### dp.Propagator{...} ###
A Propagator constructor which takes key-value arguments:
 
  * `loss` is a [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion) which the Model output will need to evaluate or minimize.
  * `callback` is a user-defined function(model, report) that does things like update the `model` parameters, gather statistics, decay learning rate, etc.
  * `epoch_callback` is a user-defined function(model, report) that is called between epochs. Typically used for learning rate decay and such;
  * `sampler` is, you guessed it, a [Sampler](data.md#dp.Sampler) instance which iterates through a [DataSet](data.md#dp.DataSet). Defaults to `dp.Sampler()`
  * `observer` is an [Observer](observer.md#dp.Observer) instance that is informed when an event occurs.
  * `feedback` is a [Feedback](feedback.md#dp.Feedback) instance that takes Model input, output and targets as input to provide I/O feedback to the user or system.
  * `progress` is a boolean that, when true, displays the progress of examples seen in the epoch. Defaults to `false`.
  * `stats` is a boolean for displaying statistics. Defaults to `false`.

<a name="dp.Optimizer"/>
[]()
## Optimizer ##

Optimizes a [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) on a `train` [DataSet](data.md#dp.DataSet).

<a name="dp.Evaluator"/>
[]()
## Evaluator ##

Evaluates a Model on a `valid` or `test` DataSet.
