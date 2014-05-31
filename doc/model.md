# Models #  
 * [Model](#dp.Model)
   * [Layer](#dp.Layer)
    * [Neural](#dp.Neural)
    * [Convolution1D](#dp.Convolution1D)
   * [Container](#dp.Container)
    * [Sequential](#dp.Sequential)

<a name="dp.Model"/>
## Model ##


<a name="dp.Model.forward"/>
### [output] forward(input, carry) ###
Forward propagates an `input` [View](view.md#dp.View) to fill and return an `output` View.

`input` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut). 
The Model will call one or many [forwardGets](view.md#dp.View.forwardGet) to retrieve a Tensor in a suitable format for forward 
propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).

`carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes.
This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of [Nodes](node.md#dp.Node).

The returned `output` is a View filled using a forwardPut.

<a name="dp.Model.forward"/>
### [output] evaluate(input, carry) ###
This method is ike forward, but for evaluation purposes (valid/test).
This is useful for stochastic Modules like Dropout, which have 
different behavior for training than for evaluation. The default is to set 
`carry.evaluate = true` and to call [forward](#dp.Model.forward).

<a name="dp.Model.backward"/>
### [input] backward(output, carry) ###
Backward propagates an `output` [View](view.md#dp.View) to fill and `input` View with a gradient and return said `input`.

`output` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut) 
in this Model's [forward](#dp.Model.forward) method and subsequently filled with a [backwardPut](view.md#dp.View.backwardPut)
from the next [Node](node.md#dp.Node) in the digraph. 
The Model will call [backwardGet](view.md#dp.View.forwardGet) on the `output` to retrieve a gradient Tensor in a suitable format for forward 
propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).

`carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes.
This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of Nodes.

The returned `input` is a View filled using a backwardPut. It is the same View that was passed to the previous call to this Model's [forward](#dp.Model.forward).
