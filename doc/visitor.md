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

<a name="dp.Visitor"/>
## Visitor ##
Abstract class for visiting a composite struture of [Models](model.md#dp.Model) 
( in particular [Layers](model.md#dp.Layer)) and modifying their states
(see [visitor design pattern](https://en.wikipedia.org/wiki/Visitor_pattern)).
Concrete Visitors should try to access a Layer method assigned to 
each visitor (if it exists). This allows models to implement their own Visitor specifics.

### dp.Visitor{name[,...]} ###
Constructrs a Visitor. Arguments should be specified as key-value pairs:
 * `name` is a string identifying the Visitor in reports. Usually this is hardcoded by the concrete sub-class;
 * `include` is a optional table where only the Models having a true value for the member named in this table are visited, unless the member is also listed in the exclude table, in which case it is not visited. If include is empty, all models are included, unless specified in the `exclude` list;
 * `exclude` is an optional table where models having a member named in this table are notvisited, even if the member is in the include table, i.e. `exclude` has precedence over `include`;
 * `observer` is an optional [Observer](observer.md#dp.Observer) that is notified when an event occurs;

<a name="dp.Learn"/>
## Learn ##
Updates the parameters of parameterized models using backward propagated gradients and learning rate(s)
Can utilize model-local learning rate scales which scale the global learning rate for that particulaar [Layer](model.md#dp.Layer).

<a name="dp.MaxNorm"/>
## MaxNorm ##
Hard constraint on the upper bound of the norm of output and/or input neuron weights (in a weight matrix). 
Has a regularization effect analogous to [WeightDecay](visitor.md#dp.WeightDecay), but with easier to optimize 
hyper-parameters. Quite useful with [ReLUs](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.ReLU). 
Should occur after [Learn](#dp.Learn) in the [VisitorChain](#dp.VisitorChain). 
Uses the C/CUDA optimized [torch.renorm](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.renorm) function.

<a name="dp.WeightDecay"/>
## WeightDecay ##
Decays the weight of the visited parameterized models.

<a name="dp.Momentum"/>
## Momentum ##
Applies momentum to parameter gradients. Should be placed before [Learn](#dp.Learn) 
in the [VisitorChain](#dp.VisitorChain)

<a name="dp.VisitorChain"/> 
## VisitorChain ##
A chain of Visitors to be executed sequentially. The order of encapsulated visitors is important.
