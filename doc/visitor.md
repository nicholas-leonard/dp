# Visitors #
Visitors visit a composite struture of Models and modify their states.
They implement the visitor design pattern which allows for double-typing:
 * [Visitor](#dp.Visitor) : abstract class inherited by all Visitors;
 * [VisitorChain](#dp.VisitorChain) : a chain of visitors where the order is important;
 * [Learn](#dp.Learn) : updates Model parameters;
 * [MaxNorm](#dp.MaxNorm) : regularization using hard constraint on max norm of weights;
 * [WeightDecay](#dp.WeightDecay) : regularization using soft constraint on norm of weights;
 * [Momentum](#dp.Momentum) : adds momentum to parameter gradients.

<a name="dp.Visitor"/>
## Visitor ##

<a name="dp.VisitorChain"/> 
## VisitorChain ##

<a name="dp.Learn"/>
## Learn ##

<a name="dp.MaxNorm"/>
## MaxNorm ##

<a name="dp.WeightDecay"/>
## WeightDecay ##

<a name="dp.Momentum"/>
## Momentum ##
