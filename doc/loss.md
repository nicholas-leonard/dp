# Cut your Losses #
Losses adapt [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion). It is very easy to build a Loss, as it often only involves wrapping a [nn](https://github.com/torch/nn/blob/master/README.md) Criterion by defining a new Loss constructor. The dp package currently supports the following Losses:
 
  * [Loss](#dp.Loss) : abstract class for measuring loss and computing gradient w.r.t. loss;
    * [Criterion](#dp.Criterion) : generic nn.Criterion adapter;
    * [NLL](#dp.NLL) : adapts the [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion);
    * [TreeNLL](#dp.TreeNLL) : used with the SoftMaxTree Model.
    * [KLDivergence](#dp.KLDivergence) : adapts [DistKLDivCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#distkldivcriterion).

<a name="dp.Loss"/>
[]()
## Loss ##
Loss is the abstract class used for wrapping Criterions. It provides default behaviour that should make it very easy to wrap any Criterion. In most cases, you need only define a constructor. Along with [Model](model.md#dp.Model), Loss shared the [Node](node.md#dp.Node) base class.

<a name="dp.Loss.__init"/>
[]()
### dp.Loss{input_view, target_view [,...]} ###
Loss constructor. Arguments should be specified as key-value pairs. Other then the following arguments, those specified in Node also apply:
 
  * `input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc.  This is usually hardcoded for each sub-class.
  * `target_view` is a string specifying the `view` of the `target` [View](view.md#dp.View) like _bt_, _b_, etc. This is usually hardcoded for each sub-class.
  * `target_type` is a string identifying the type of target Tensors. It defaults to `torch.getdefaulttensortype()` which is usually _torch.DoubleTensor_.
  * `input_module` is an optional [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) to use on the inputs. A good example is when the output of the Model uses a [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#softmax) yet the wrapped Criterion, like [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion), expects this output to be passed through a [Log](https://github.com/torch/nn/blob/master/Log.lua).
  * `size_average` is boolean that when set to true (the default) expects the wrapped Criterion to output a loss averaged by the size of the input. This is common in Criterions with a default `sizeAverage=true` attribute. More generally, it the `_forward/_evaluate` methods set the `loss` attribute with a value averaging the loss of the batch, this value should be set to true. It only affects how the `loss` values are accumulated in the `_stats` attribute for reporting.

<a name="dp.Loss.forward"/>
[]()
### [loss, carry] forward(input, target, carry) ###
Measures loss of an `input` [View](view.md#dp.View) with respect to a `target` View.

  * `input` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut). The Loss will call one or many [forwardGets](view.md#dp.View.forwardGet) to retrieve a Tensor in a suitable format for (in most cases) calling the adapted [Criterion:forward](https://github.com/torch/nn/blob/master/doc/criterion.md#output-forwardinput-target).
  * `carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes. This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of [Nodes](node.md#dp.Node).

The returned `loss` is a scalar measuring the loss for the particular `input` and `target` mini-batch of examples.

<a name="dp.Loss.evaluate"/>
[]()
### [loss, carry] evaluate(input, target, carry) ###
This method is like [forward](#dp.Loss.forward), but for evaluation purposes (valid/test).
This is useful for inserting stochastic Modules like Dropout, which have 
different behavior for training than for evaluation. The default is to set 
`carry.evaluate = true` and to call [forward](#dp.Loss.forward).

<a name="dp.Loss.backward"/>
[]()
### [input, carry] backward(input, target, carry) ###
Backward propagates an `output` [View](view.md#dp.View) to fill and `input` View with a gradient and return said `input`.
 
  * `input` is a [View](view.md#dp.View) that will be filled with gradients by this method via a call to  [backwardPut](view.md#dp.View.backwardPut).
  * `carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes. This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of Nodes.

The returned `input` is a View filled using a backwardPut. It is the same View that was passed to the previous call to this object's [forward](#dp.Loss.forward) method.

<a name='dp.Loss.inputAct'/>
[]()
### [act] inputAct() ###
Returns the result of a [forwardGet](view.md#dp.View.forwardGet) on the Loss's `input` 
using its `input_view` and `input_type` attributes.

<a name='dp.Loss.inputGrad'/>
[]()
### inputGrad(input_grad) ###
Sets the Loss's `input` gradient by calling its [backwardPut](view.md#dp.View.backwardPut) using its `input_view` 
and the provided `input_grad` Tensor.

<a name='dp.Loss.targetAct'/>
[]()
### targetAct(output_act) ###
Returns the result of a [forwardGet](view.md#dp.View.forwardGet) on the Loss's `target` 
using its `target_view` and `target_type` attributes.

<a name='dp.Loss.avgError'/>
[]()
### [error] Loss:avgError() ###
Returns the average error (or loss) over all examples seen since the last call to [zeroStatistics](node.md#dp.Node.zeroStatistics) (or [doneEpoch](node.md#dp.Node.zeroStatistics) for that matter).
The value returned also depends on the `size_average` [constructor](#dp.Loss.__init) argument having been set right (for most cases, the default is fine).

<a name="dp.Criterion"/>
[]()
## Criterion ##
A generic [nn.Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion) adapter. Not to be confused with nn.Criterion. Use this to quickly wrap a nn.Criterion into a [Loss](#dp.Loss). 
For all intents and purposes, it should do a great 
job of integrating your existing Criterions into dp. Just wrap them using this Loss. 

<a name="dp.Criterion.__init"/>
[]()
### dp.Criterion{criterion[, input_view, target_view]} ###
Criterion constructor. Other then the following 
arguments, those specified in [Loss](#dp.Loss.__init) also apply:
 
  * `criteiron` is a nn.Criterion instance.
  * `input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc. Defaults to `default` axis view.
  * `target_view` is a string specifying the `view` of the `target` [View](view.md#dp.View) like _bt_, _b_, etc. Defaults to `default` axis view.

<a name="dp.NLL"/>
[]()
## NLL ##
Adapts [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion) for measuring the Negative Log Likelihood (NLL) and computing its gradient.


<a name="dp.TreeNLL"/>
[]()
## TreeNLL ##
Negative Log Likelihood (NLL) for Models ending with a SoftmaxTree. Used for maximizing the likelihood of SoftmaxTree Model outputs. SoftmaxTree outputs a column tensor representing the log likelihood of each target in the batch. Thus SoftmaxTree requires the targets. So this Loss only computes the negative of those outputs, as well as its corresponding gradients. This class doesn't wrap any [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion).

<a name="dp.KLDivergence"/>
[]()
## KLDivergence ##
Adapts [DistKLDivCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#distkldivcriterion) for measuring the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between an input and target distribution. Unlike [NLL](#dp.NLL), this Criterion acts in a purely element-wise fashion thereby accepting inputs and targets of any size, as long as the number of elements is the same.
