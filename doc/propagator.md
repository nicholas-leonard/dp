# Propagators #
These propagate Batches sampled from a DataSet using a Sampler through a 
[Model](model.md#dp.Model) in order to evaluate a Loss, provide Feedback or
train the model :
 * [Propagator](#dp.Propagator) : abstract class;
   * [Optimizer](#dp.Optimizer) : optimizes a [Model](model.md#dp.Model) on a `train` [DataSet](data.md#dp.DataSet);
   * [Evaluator](#dp.Evaluator) : evaluates a Model on a `valid` or `test` DataSet.

<a name="dp.Propagator"/>
## Propagator ##
Abstract Class for propagating a sampling distribution (a Sampler) through a 
[Model](model.md#dp.Model). A Propagator can be sub-classed to build task-tailored 
training or evaluation algorithms. If such a propagator can be further reduced 
to reusable components, it can also be refactored into [Visitors](visitor.md), 
[Observers](observer.md), and so on. 

<a name="dp.Propagator.__init"/>
### dp.Propagator{...} ###
A Propagator constructor which takes key-value arguments:
 * `loss` is a [Loss](loss.md#dp.Loss) which the Model output will need to evaluate or minimize.
 * `visitor` is a [Visitor](visitor.md#dp.Visitor) instance which visits component Models at the end of each batch propagation to perform parameter updates and/or gather statistics, etc.
 * `sampler` is, you guessed it, a [Sampler](sampler.md#dp.Sampler) instance which iterates through a [DataSet](data.md#dp.DataSet). Defaults to `dp.Sampler()`
 * `observer` is an [Observer](observer.md#dp.Observer) instance that is informed when an event occurs.
 * `feedback` is a [Feedback](feedback.md#dp.Feedback) instance that takes Model input, output and targets as input to provide I/O feedback to the user or system.
 * `progress` is a boolean that, when true, displays the progress of examples seen in the epoch. Defaults to `false`.
 * `stats` is a boolean for displaying statistics. Defaults to `false`.

<a name="dp.Optimizer"/>
## Optimizer ##

<a name="dp.Evaluator"/>
## Evaluator ##
