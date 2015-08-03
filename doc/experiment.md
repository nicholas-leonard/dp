# Experiment #
An experiment propagates DataSets (encapsulated by a DataSource) 
through a Model via method [run](#dp.Experiment.run). The specifics 
of such propagations are handled by Propagators. The propagation of 
a DataSet is called an epoch. At the end of each epoch, a monitoring
step is performed where reports are generated for all Observers.

The Experiment keeps a log of the report of the experiment after 
every epoch. This is done by calling the [report](#dp.Experiment.report) method of 
every contained object, except Observers. The report is a read-only 
table that is passed to Observers along with the Mediator through 
Channels for which they are Subscribers. The report is also passed 
to sub-objects during propagation in case they need to act upon 
data found in other branches of the experiment tree.

<a name="dp.Experiment.__init"/>
[]()
## dp.Experiment{...} ##
An Experiment constructor which takes key-value arguments:

  * `id` is an ObjectID uniquely identifying the experiment. Defaults to using `dp.uniqueID()`.
  * `model` is a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) instance shared by all Propagators.
  * `optimizer` is an [Optimizer](propagator.md#dp.Optimizer) instance used for propagating the train set.
  * `validator` is an [Evaluator](propagator.md#dp.Evaluator) instance used for propagating the valid set. 
  * `tester` is an [Evaluator](propagator.md#dp.Evaluator) instance used for propagating the test set. 
  * `observer` is an [Observer](observer.md#dp.Observer) instance used for extending the subject Experiment. 
  * `random_seed` is a number used to initialize the random number generator. Defouts to `7`.
  * `epoch` at which to start the experiment. This is useful reusing trained Models in new Experiments. Defaults to `0`.
  * `mediator` is a Mediator, a Singleton, used for inter-object communication. Defaults to `Mediator()`.
  * `overwrite` is a boolead with a default value of `false` used to overwrite existing values. For example, if a datasource is provided, and optimizer is already nitialized with a dataset, and overwrite is true, then optimizer would be setup with `datasource:trainSet()`.
  * `max_epoch` sets the maximum number of epochs allocated to the experiment. Defaults to `1000`.
  * `description` a short description of the experiment.

The only mandatory arguments are `model` and at least one `optimizer`, `validator` or `tester`.

<a name="dp.Experiment.run"></a>
## run(datasource) ##
This method loops through the propagators until a doneExperiment is 
received or experiment reaches `max_epoch` epochs. The `datasource` 
is provided as an argument here, as opposed to being passed to the 
constructor to keep it from being serialized with the Experiment. 

<a name="dp.Experiment.includeTarget"></a>
## includeTarget(mode) ##
When mode is `true` (the default), the targets will be included in 
the input to every `model`. Such that the input will have the form :
```lua
input = {input, target}
```
The default behavior for the Experiment is to not include the targets in the input. 
This is useful for modules like [SoftMaxTree](https://github.com/clementfarabet/lua---nnx#nnx.SoftMaxTree) 
which require the target be fed in as well as the input.

<a name="dp.Experiment.report"/>
[]()
## [report] report() ##
This method returns a `report` table using for monitoring the components
of the Experiment. This report includes reports 
(in attributes of the same name) for the `optimizer`, 
`validator`, `tester` and `model`. It also includes attributes for 
the current `epoch`, `random_seed` and `id`.

<a name="dp.Experiment.verbose"></a>
## verbose([on]) ##
Toggle the verbosity of all objects in the experiment. When `on` is 
true (the default), objects will print messages.

<a name="dp.Experiment.silent"></a>
## silent() ##
Calls `self:verbose(false)`.
