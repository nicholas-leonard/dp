# Losses #
Losses adapt [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion). Like [Model](model.md#model), a Loss is a [Node](node.md#node-1). It is very easy to build a Loss, as it often only involves wrapping a [nn](https://github.com/torch/nn/blob/master/README.md) Criterion. The dp package currently supports the following Losses:
 * [Loss](#dp.Loss)
 * [NLL](#dp.NLL) : adapts the ClassNLLCriterion;
 * [TreeNLL](#dp.TreeNLL) : used with the SoftMaxTree Model.

<a name="dp.Loss"/>
## Loss ##
Loss is the abstract class used for wrapping Criterions. It provides default behaviour that should make it very easy to wrap any Criterion. In most cases, you need only define a constructor. Along with [Model](model.md#dp.Model), Loss shared the [Node](node.md#dp.Node) base class.

<a name="dp.Loss.__init"/>
### dp.Loss{input_view, target_view [,...]} ###
Node constructor. Arguments should be specified as key-value pairs:
 * `input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc.  This is usually hardcoded for each sub-class.
 * `target_view` is a string specifying the `view` of the `target` [View](view.md#dp.View) like _bt_, _b_, etc. This is usually hardcoded for each sub-class.
 * `target_type` is a string identifying the type of target Tensors. It defaults to `torch.getdefaulttensortype()` which is usually _torch.DoubleTensor_.
 'type of target tensors'},
 * `input_module` is an optional [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) to use on the inputs. A good example is when the output of the Model uses a [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#softmax) yet the wrapped Criterion, like [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion), expects this output to be passed through a [Log](https://github.com/torch/nn/blob/master/Log.lua).
 * `size_average` is boolean that when set to true (the default) expects the wrapped Criterion to output a loss averaged by the size of the input. This is common in Criterions with a default `sizeAverage=true` attribute. More generally, it the `_forward/_evaluate` methods set the `loss` attribute with a value averaging the loss of the batch, this value should be set to true. It only affects how the `loss` values are accumulated in the `_stats` attribute for reporting.

<a name="dp.NLL"/>
## NLL ##


<a name="dp.TreeNLL"/>
## TreeNLL ##
