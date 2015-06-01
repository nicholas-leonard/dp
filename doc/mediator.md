# Mediate (or Bend the Rules) #
The _dp_ library can be very rigid in how the interactions between different objects 
are regulated (e.g. how [DataSet](data.md#dp.DataSet), 
[Propagator](propagator.md#dp.Propagator), [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module)
and [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion) all work together like clockwork). 
Yet when doing research and development in the field of deep learning, 
we often need to build models that require non-standard interactions.
While the library cannot provide rigid interaction patterns for every research use case,
it provides different flexible alternatives. In this section we examine the Mediator singleton
and its supporting classes that implement the 
[Publish–subscribe pattern](http://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern):
 
  * [Mediator](#dp.Mediator) : singleton that implements listen-notify pattern;
  * [Channel](#dp.Channel) : can be published and subscribed to;
  * [Subscriber](#dp.Subscriber) : encapsulates a listening object and callback method;

<a name="dp.Mediator"/>
[]()
## Mediator ##
A singleton referenced by most objects directly or indirectly 
encapsulated by an [Experiment](experiment.md#dp.Experiment).

<a name="dp.Channel"/>
[]()
## Channel ##
Used internally by [Mediator](#dp.Mediator). Can be published and subscribed to.

<a name="dp.Subscriber"/>
[]()
## Subscriber ##
Used internally by [Mediator](#dp.Mediator). Holds a listening object (or *subscriber*) which will be called
back via its `func_name` method name. Since no functions are being 
pointed to directly, the object can be serialized.
