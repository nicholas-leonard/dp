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

<a name="dp.Node.forward"/>
### [output] forward(input, carry) ###
Forwards an `input` [View](node.md#dp.View) to generate an `output` View.

`input` is a [View](node.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut). 
The Node will call one or many [forwardGets](view.md#dp.View.forwardGet) to retrieve a Tensor in a suitable format for forward 
propagation through the Node's [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) 
or [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion).

`carry` is a table is carried throughout the graph. A Node can modify it, but should avoid deleting attributes.
This is useful when you want to forward information to a later Node in the graph seperated by an unknown number of Nodes.

In the case of a Model, the returned `output` is a View filled using a forwardPut. In the case of a Loss, it is a scalar 
measure of the loss.

