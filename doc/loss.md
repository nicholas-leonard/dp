# Losses #
Losses adapt [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion). Like [Model](model.md#model), a Loss is a [Node](node.md#node-1). It is very easy to build a Loss, as it often only involves wrapping a [nn](https://github.com/torch/nn/blob/master/README.md) Criterion. The dp package currently supports the following Losses:
 * [NLL](#NLL) : adapts the ClassNLLCriterion;
 * [TreeNLL](#TreeNLL) : used with the SoftMaxTree Model.

## Loss ##

## NLL ##

## TreeNLL ##
