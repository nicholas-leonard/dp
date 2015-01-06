# Visitors #
Visitors visit a composite struture of Models and modify their states.
They implement the [visitor design pattern](https://en.wikipedia.org/wiki/Visitor_pattern) 
which allows for separating a training algorithm from a structure of [Models](model.md#dp.Models) 
on which it operates (in effect, implementing [double-dispatch](https://en.wikipedia.org/wiki/Double_dispatch)):

  * [Visitor](#dp.Visitor) : abstract class inherited by all Visitors;
    * [Learn](#dp.Learn) : updates Model parameters;
    * [MaxNorm](#dp.MaxNorm) : regularization using hard constraint on max norm of weights;
    * [WeightDecay](#dp.WeightDecay) : regularization using soft constraint on norm of weights;
    * [Momentum](#dp.Momentum) : adds momentum to parameter gradients.
    * [VisitorChain](#dp.VisitorChain) : a chain of visitors where the order is important;
      * [RecurrentVisitorChain](#dp.RecurrentVisitorChain) : used by recurrent Models to visit sequences of batches;

<a name="dp.Visitor"/>
[]()
## Visitor ##
Abstract class for visiting a composite struture of [Models](model.md#dp.Model) 
(in particular [Layers](model.md#dp.Layer)) and modifying their states
(see [visitor design pattern](https://en.wikipedia.org/wiki/Visitor_pattern)).
Concrete Visitors should try to access a Layer method assigned to 
each visitor (if it exists). This allows models to implement their own Visitor specifics.

<a name='dp.Visitor.__init'/>
[]()
### dp.Visitor{name[,...]} ###
Constructs a Visitor. Arguments should be specified as key-value pairs:

  * `name` is a string identifying the Visitor in reports. Usually this is hardcoded by the concrete sub-class;
  * `zero_grads` is a boolean. When true (the default), the visitor calls [zeroGradParameters](model.md#dp.Model.zeroGradParameters) on each model after it has been visited. Should be true for the root vistior. Note that the [VisitorChain](#dp.VisitorChain) sets this attribute to false for all its component visitors, such that this method is called only once.
  * `include` is a optional table where only the Models having a true value for the member named in this table are visited, unless the member is also listed in the exclude table, in which case it is not visited. If include is empty, all models are included, unless specified in the `exclude` list;
  * `exclude` is an optional table where models having a member named in this table are notvisited, even if the member is in the include table, i.e. `exclude` has precedence over `include`;
  * `observer` is an optional [Observer](observer.md#dp.Observer) that is notified when an event occurs;
  * `verbose` is a boolean. When true (the default), it can prints messages to `stdout`.

<a name="dp.Learn"/>
[]()
## Learn ##
Updates the parameters of parameterized models using backward 
propagated gradients and learning rate(s)
Can utilize model-local learning rate scales which scale the 
global learning rate for that particular [Layer](model.md#dp.Layer).
These are specified using the `mvstate` argument of the 
[Model constructor](model.md#dp.Model.__init). For example:
```lua
layer = dp.Neural{
   mvstate = {learn_scale = 0.1},
   [...]
}
```
In which case, the learning rate for that layer will be `learning_rate*learn_scale`.

<a name="dp.Learn.__init"/>
### dp.Learn{learning_rate} ###
Constructs a Learn Visitor. Arguments should be specified as key-value pairs. 
Other then the following argument, those specified in [Visitor](#dp.Visitor.__init) also apply.

  * `learning_rate` is a number specifying the learning rate of parameters updates.

<a name="dp.MaxNorm"/>
[]()
## MaxNorm ##
Hard constraint on the upper bound of the norm of output and/or input neuron weights (in a weight matrix). 
Has a regularization effect analogous to [WeightDecay](visitor.md#dp.WeightDecay), but with easier to optimize 
hyper-parameters. Quite useful with [ReLUs](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.ReLU). 
Should occur after [Learn](#dp.Learn) in the [VisitorChain](#dp.VisitorChain). 
Uses the C/CUDA optimized [torch.renorm](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.renorm) function.

<a name="dp.MaxNorm.__init"/>
[]()
### dp.MaxNorm{...} ###
Constructs a MaxNorm Visitor. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Visitor](#dp.Visitor.__init) also apply.

  * `max_out_norm` specifies the maximum norm of output neuron weights. If not specified, this constraint is ignored. Defaults to 1.
  * `max_in_norm` specifies the maximum norm of input neuron weights. If not specified, this constraint is ignored. Default is to ignore.
  * `period` specifies the periodicity of application of the constraint. Every `period` batches, the norm is constrained. Defaults to 1.

<a name="dp.WeightDecay"/>
[]()
## WeightDecay ##
Decays the weight of the visited parameterized models.

<a name="dp.Momentum"/>
[]()
## Momentum ##
Applies momentum to parameter gradients. Should be placed before [Learn](#dp.Learn) 
in the [VisitorChain](#dp.VisitorChain)

<a name="dp.VisitorChain"/> 
[]()
## VisitorChain ##
A chain of Visitors to be executed sequentially. 
The order of encapsulated visitors is important.

<a name="dp.VisitorChain.__init"/>
[]()
### dp.VisitorChain{visitors} ###
Constructs a VisitorChain Visitor. Arguments should be specified as key-value pairs. 
Other then the following argument, those specified in [Visitor](#dp.Visitor.__init) also apply.

  * `visitors` is a sequence (a table) of visitors to apply to visited models.

<a name="dp.RecurrentVisitorChain"/>
[]()
## RecurrentVisitorChain ##
A composite chain of visitors used to visit recurrent models. 
Subscribes to Mediator channel `"doneSequence"`. When this channel is notified, 
the next time this visitor is [accepted](model.md#dp.Model.accept) by the model, 
the contained visitors will update (visit) it. 
Otherwise, the model is visited every `visit_interval` epochs.
Used for [Recurrent Neural Network Language Models](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua).

<a name="dp.RecurrentVisitorChain.__init"/>
[]()
### dp.RecurrentVisitorChain{...} ###
Constructs a RecurrentVisitorChain Visitor. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Visitor](#dp.Visitor.__init) also apply.

  * `visit_interval` controls the frequency of updates. The [Model](model.md#dp.Model) is updated (visited) every `visit_interval` epochs. However, if [Channel](mediator.md#dp.Channel) `"doneSequence"` was notified, the Model will be updated at the end of the current epoch.
  * `force_forget` is boolean. When true, it forces recurrent models to [forget](https://github.com/clementfarabet/lua---nnx#nnx.Recurrent) after each update. Defaults to false.
