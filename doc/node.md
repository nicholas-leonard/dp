# Node #
The Abstract class [Node](#dp.Node) attempts to factor-out the commonalities of 
Losses and Models.

<a name="dp.Node"/>
## Node ##
Abstract class inherited by Model and Loss.
Forward and backward propagates representations (Tensors or Tables).
Nodes are joined by [Views](node.md#dp.View). In the context of a digraph, 
we can think of Nodes as, you guessed it, nodes, and Views as sets of outgoing arrows (one input, multiple output).

<a name="dp.Node.__init"/>
### dp.Node{[input_type, output_type, module_type]} ###
Node constructor. Arguments should be specified as key-value pairs. 

`input_type` is a string identifying the type of input activation and gradient Tensors. It defaults to 
`torch.getdefaulttensortype()` which is usually _torch.DoubleTensor_.

`output_type` is a string identifying the type of output activation and gradient Tensors. In the case of 
Loss, it identifies the type of the targets. It also defaults to `torch.getdefaulttensortype()`.

`module_type` is a string identifying the type of [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) 
or [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion) used within the Node. It also defaults to 
`torch.getdefaulttensortype()`.

<a name="dp.Node.setup"/>
### setup{mediator, id} ###
Post-initialization setup method usually initiated (indirectly called) by Experiment. 
Arguments should be specified as key-value pairs. 

`mediator` is a Mediator which allows Nodes to signal other object of events.

`id` is an ObjectID that uniquely identifies the Node. This is useful for identifying 
Nodes in reports, and uniquely identifying objects between experiments.

<a name="dp.Node.report"/>
### [report] report() ###
Returns a report of the Node's state and progress for later analysis. 
If statistics were being gathered, this is the time to report them.
Expect report to be called at least every epoch.
Observers can also use data found in the reports generated every epoch for annealing 
values, early-stopping, etc. Propagators and Experiments also have reports such that these 
call the reports of their internal objects to generate a sub-tree of the final report.

<a name="dp.Node.zeroStatistics"/>
### zeroStatistics() ###
Zeros statistics between epochs.

<a name="dp.Node.updateStatistics"/>
### updateStatistics(carry) ###
Should only be called by forward or evaluate (once per batch). Uses 
its internal state and metadata found in `carry` table (see [Model](model.md#dp.Model.forward) and [Loss](loss.md#dp.Loss.forward)), 
to upsate its internal `self._stats` table of statistics.

<a name="dp.Node.doneBatch"/>
### doneBatch(...) ###
This should be called on completion of each batch propagation. It can reset its internal state for the next batch, accumulate statistics on the batch, 
zero parameter gradients, etc.

<a name="dp.Node.doneEpoch"/>
### doneEpoch([report,] ...) ###
This should be called between each epoch to to [zeroStatistics](#dp.Node.zeroStatistics) and such. 
Altough seldom used in practive, the optional `report` argument contains a complete report of the epoch. 
