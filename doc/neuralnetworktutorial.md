<a name="NeuralNetworkTutorial"/>
[]()
# Neural Network Tutorial #

We begin with a simple [neural network example](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork_tutorial.lua). The first line loads 
the __dp__ package, whose first matter of business is to load its dependencies (see [init.lua](https://github.com/nicholas-leonard/dp/blob/master/init.lua)):
```lua
require 'dp'
```
Note : package [Moses](https://github.com/Yonaba/Moses/blob/master/docs/moses.md) is imported as `_`. So `_` shouldn't be used for dummy variables. 
Instead use the much more annoying `__`, or whatnot. 

Lets define some hyper-parameters and store them into a table. We will need them later:
```lua
--[[hyperparameters]]--
opt = {
   nHidden = 100, --number of hidden units
   learningRate = 0.1, --training learning rate
   momentum = 0.9, --momentum factor to use for training
   maxOutNorm = 1, --maximum norm allowed for output neuron weights
   batchSize = 128, --number of examples per mini-batch
   maxTries = 100, --maximum number of epochs without reduction in validation error.
   maxEpoch = 1000 --maximum number of epochs of training
}
```
## DataSource and Preprocess ##
We intend to build and train a neural network so we need some data, 
which we encapsulate in a [DataSource](data.md#dp.DataSource)
object. __dp__ provides the option of training on different datasets, 
notably [MNIST](data.md#dp.Mnist), [NotMNIST](data.md#dp.NotMnist), 
[CIFAR-10](data.md#dp.Cifar10) or [CIFAR-100](data.md#dp.Cifar100), but for this
tutorial we will be using the archetypal MNIST (don't leave home without it):
```lua
--[[data]]--
datasource = dp.Mnist{input_preprocess = dp.Standardize()}
```
A DataSource contains up to three [DataSets](data.md#dp.DataSet): 
`train`, `valid` and `test`. The first is for training the model. 
The second is used for [early-stopping](observer.md#dp.EarlyStopper) and cross-validation.
The third is used for publishing papers and comparing results across different models.
  
Although not really necessary, we [Standardize](preprocess.md#dp.Standardize) 
the datasource, which subtracts the mean and divides 
by the standard deviation. Both statistics (mean and standard deviation) are 
measured on the `train` set only. This is a common pattern when preprocessing data. 
When statistics need to be measured across different examples 
(as in [ZCA](preprocess.md#dp.ZCA) and [LecunLCN](preprocess.md#dp.LeCunLCN) preprocesses), 
we fit the preprocessor on the `train` set and apply it to all sets (`train`, `valid` and `test`). 
However, some preprocesses require that statistics be measured
only on each example (as in [global constrast normalization](preprocess.md#dp.GCN)). 

## Model of Modules ##

Ok so we have a DataSource, now we need a model. Let's build a 
multi-layer perceptron (MLP) with two parameterized non-linear layers:
```lua
--[[Model]]--

model = nn.Sequential():extend(
   nn.Convert(ds:ioShapes(), 'bf'), -- to batchSize x nFeature (also type converts)
   nn.Linear(ds:featureSize(), opt.nHidden), 
   nn.Tanh(),
   nn.Linear(opt.nHidden, #(ds:classes())),
   nn.LogSoftMax()
)
```
Both layers are defined using a [Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear),
which contains the parameters that will be learned, followed by a non-linear transfer function like 
[Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh) (for the hidden neurons) 
and [LogSoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.LogSoftMax) (for the output neurons).
The latter might seem odd (why not use [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.SoftMax) instead?), 
but the [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion) only works 
with LogSoftMax (or with SoftMax + [Log](https://github.com/torch/nn/blob/master/Log.lua)).

The Linear modules are constructed using 2 arguments, `inputSize` (number of input units) and `outputSize` (number of output units).
For the first layer, the `inputSize` is the number of features in the input image. 
In our case, that is `1x28x28=784`, which is what `ds:featureSize()` will return.

Now for the odd looking `nn.Convert` Module. It has two purposes. First, 
whatever type of Tensor received, it will output the type of Tensor used by the Module.
Second, it can convert from different Tensor shapes. The input shape of a typical image is 
*bchw*, short for  *batch*, *color/channel*, *height*, *width*. Modules like 
[SpatialConvolution]() and [SpatialMaxPooling]() expect this type of input. Our MLP, on the other 
hand, expects an input of shape *bf*, short for *batch*, *feature*. Its a pretty simple conversion 
actually, all you need to do is flatten the *chw* dimensions to a single *f* dimension (in this case, of size 784). 

For those not familiar with the nn package, all the `nn.*` in the above snippet of code 
are [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) subclasses. 
This is true even for the [Sequential](https://github.com/torch/nn/blob/master/containers.md#nn.Sequential).
Although the latter is special as it is a [Container](https://github.com/torch/nn/blob/master/containers.md#nn.Container) of other Modules.

## Propagator ##

Next we initialize some [Propagators](propagator.md#dp.Propagator). 
Each such Propagator will propagate examples from a different [DataSet](data.md#dp.DataSet).
[Samplers](data.md#dp.Sampler) iterate over DataSets to 
generate [Batches](data.md#dp.Batch) of examples (inputs and targets) to propagate through the `model`:
```lua
--[[Propagators]]--

train = dp.Optimizer{
   loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Convert()),
   callback = function(model, report) 
      -- the ordering here is important
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   feedback = dp.Confusion(),  
   sampler = dp.Sampler()
}
test = dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler()
}

```
For this example, we use an [Optimizer](propagator.md#dp.Optimizer) for the training DataSet,
and two [Evaluators](propagator.md#dp.Evaluator), one for cross-validation 
and another for testing. 

### Sampler ###
The Evaluators use a simple Sampler which 
iterates sequentially through the DataSet. On the other hand, the Optimizer 
uses a [ShuffleSampler](data.md#dp.SuffleSampler). This Sampler
shuffles the (indices of a) DataSet before each pass over all examples in a DataSet. 
This shuffling is useful for training since the model 
must learn from varying sequences of batches through the DataSet, 
which makes the training algorithm more stochastic (subject to the constraint that 
each example is presented once and only once per epoch).

### Loss ###
Each Propagator can also specify a `loss` for training or evaluation. This argument is 
only mandatory for the Optimizer, as it is required for [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
If you have previously used the [nn](https://github.com/torch/nn/blob/master/README.md) package, 
there is nothing new here. The loss is a [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion). 
Each example has a single target class and our Model output is LogSoftMax so 
we use a [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion).
The criterion is wrapped in [ModuleCriterion](https://github.com/nicholas-leonard/dpnn/blob/master/README.md#nn.ModuleCriterion).
This isn't really useful to this particular script as it is never cast to a different type

### Feedback ###
The `feedback` parameter is used to provide us with, you guessed it, feedback (like performance measures and
statistics after each epoch). We use [Confusion](feedback.md#dp.Confusion), which is a wrapper 
for the [optim](https://github.com/torch/optim/blob/master/README.md) package's 
[ConfusionMatrix](https://github.com/torch/optim/blob/master/ConfusionMatrix.lua).
While our Loss measures the Negative Log-Likelihood (NLL) of the Model 
on different DataSets, our [Feedback](feedback.md#feedback) 
measures classification accuracy (which is what we will use for 
early-stopping and comparing our model to the state-of-the-art).

### Visitor ###
Since the [Optimizer](propagator.md#dp.Optimizer) is used to train the Model on a DataSet, 
we need to specify some Visitors to update its [parameters](model.md#dp.Model.parameters). 
We want to update the Model by sequentially applying the following visitors: 

  1. [Momentum](visitor.md#dp.Momentum) : updates parameter gradients using a factored mixture of current and previous gradients.
  2. [Learn](visitor.md#dp.Learn) : updates the parameters using the gradients and a learning rate.
  3. [MaxNorm](visitor.md#dp.MaxNorm) : updates output or input neuron weights (in this case, output) so that they have a norm less than or equal to a specified value.

The only mandatory Visitor is the second one (Learn), which does the actual parameter updates. 
The first is the well-known Momentum. 
The last (MaxNorm) is the lesser-known hard constraint on the norm of output or input neuron weights 
(see [Hinton 2012](http://arxiv.org/pdf/1207.0580v1.pdf)), which acts as a regularizer. You could also
replace it with a more classic regularizer like [WeightDecay](visitor.md#dp.WeightDecay), in which case you 
would have to put it *before* the Learn visitor.

Finally, we have the Optimizer switch on its `progress` bar so we 
can monitor its progress during training. 

## Experiment ##
Now its time to put this all togetherto form an [Experiment](experiment.md):
```lua
--[[Experiment]]--
xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}
```
### Observer ###
The Experiment can be initialized with a list of [Observers](observer.md#dp.Observer). The 
order is not important. Observers listen to mediator [Channels](mediator.md#dp.Channel). The Mediator 
calls them back when certain events occur. In particular, they may listen to the _doneEpoch_
Channel to receive a report from the Experiment after each epoch. A report is nothing more than 
a hierarchy of tables. After each epoch, the component objects of the Experiment (except Observers) 
can each submit a report to its composite parent thereby forming a tree of reports. The Observers can analyse 
these and modify the components which they are assigned to (in this case, Experiment). 
Observers may be attached to Experiments, Propagators, Visitors, etc. 

#### FileLogger ####
Here we use a simple [FileLogger](observer.md#dp.FileLogger) which will 
store serialized reports in a simple text file for later use. Each experiment has a unique ID which is 
included in the corresponding reports, thus allowing the FileLogger to name its file appropriately. 

#### EarlyStopper ####
The [EarlyStopper](observer.md#dp.EarlyStopper) is used for stopping the Experiment when error has not decreased, or accuracy has not 
been maximized. It also saves to disk the best version of the Experiment when it finds a new one. 
It is initialized with a channel to `maximize` or minimize (the default is to minimize). In this case, we intend 
to early-stop the experiment on a field of the report, in particular the _accuracy_ field of the 
_confusion_ table of the _feedback_ table of the `validator`. 
This `{'validator','feedback','confusion','accuracy'}` happens to measure the accuracy of the Model on the 
validation DataSet after each training epoch. So by early-stopping on this measure, we hope to find a 
Model that generalizes well. The parameter `max_epochs` indicates how many consecutive 
epochs of training can occur without finding a new best model before the experiment is signaled to stop 
by the _doneExperiment_ Mediator Channel.

## Running the Experiment ##
Once we have initialized the experiment, we need only run it on the `datasource` to begin training.
```lua
xp:run(datasource)
```
We don't initialize the Experiment with the DataSource so that we may easily 
save it to disk, thereby keeping this snapshot separate from its data 
(which shouldn't be modified by the experiment).

Let's run the [script](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork_tutorial.lua) from the cmd-line:
```
nicholas@xps:~/projects/dp$ th examples/neuralnetwork_tutorial.lua 
FileLogger: log will be written to /home/nicholas/save/xps:25044:1398320864:1/log	
xps:25044:1398320864:1:optimizer:loss avgError 0	
xps:25044:1398320864:1:validator:loss avgError 0	
xps:25044:1398320864:1:tester:loss avgError 0	
==> epoch # 1 for optimizer	
 [================================ 50000/50000 ===============================>] ETA: 0ms | Step: 0ms                              

==> epoch size = 50000 examples	
==> batch duration = 0.10882427692413 ms	
==> epoch duration = 5.4412138462067 s	
==> example speed = 9189.1260687829 examples/s	
==> batch speed = 71.790047412366 batches/s	
xps:25044:1398320864:1:optimizer:loss avgError 0.0037200363330228	
xps:25044:1398320864:1:validator:loss avgError 0.004545687570244	
xps:25044:1398320864:1:tester:loss avgError 0.0047699521407681	
xps:25044:1398320864:1:optimizer:confusion accuracy = 0.88723958333333	
xps:25044:1398320864:1:validator:confusion accuracy = 0.92788461538462	
xps:25044:1398320864:1:tester:confusion accuracy = 0.92027243589744	
==> epoch # 2 for optimizer	
 [================================ 50000/50000 ===============================>] ETA: 0ms | Step: 0ms                              

==> epoch size = 50000 examples	
==> batch duration = 0.10537392139435 ms	
==> epoch duration = 5.2686960697174 s	
==> example speed = 9490.0141018538 examples/s	
==> batch speed = 74.140735170733 batches/s	
xps:25044:1398320864:1:optimizer:loss avgError 0.0023303674656008	
xps:25044:1398320864:1:validator:loss avgError 0.0044356466501897	
xps:25044:1398320864:1:tester:loss avgError 0.0046304688698266	
xps:25044:1398320864:1:optimizer:confusion accuracy = 0.92375801282051	
xps:25044:1398320864:1:validator:confusion accuracy = 0.93129006410256	
xps:25044:1398320864:1:tester:confusion accuracy = 0.92548076923077	
==> epoch # 3 for optimizer	
 [===============................ 10112/50000 ................................] ETA: 8s540ms | Step: 0ms  
```

## Hyperoptimizing ##

Hyper-optimization is the hardest part of deep learning. 
In many ways, it can feel more like an art than a science. 
[Momentum](visitor.md#dp.Momentum) can help convergence, but it requires much more memory. 
The same is true of weight decay, as both methods require a 
copy of parameter gradients which often almost double the memory footprint of the model. 
Using [MaxNorm](visitor.md#dp.MaxNorm) and [AdaptiveLearningRate](observer.md#dp.AdaptiveLearningRate) is often better as 
experiments can gain more from the extra memory when it 
is used instead for more modeling capacity (more parameters). 
But this may be mostly applicable to large datasets like the [BillionWords](data.md#dp.BillionWords) dataset. 
The models it requires are proportionally heavy to the vocabulary size, which is `~800,000` unique words.

Anyway, rule of thumb, always start hyper-optimizing 
by seeking the highest learning rate you can afford. 
You will need to try many different experiments (hyper-parameter configurations), 
so you need them to converge fast. 
If you are worried about controlling the decay of the learning rate, 
try out the AdaptiveLearningRate Observer. 

Regularize with MaxNorm Visitor. 
A `max_out_norm` around 2 is usually a good starting point, continuing with 1, 10, 
and only try 1000000000 when out of ideas. 
You can vary the epoch sizes to divide processing time 
between evaluation and training. 
It's often best to keep the evaluation sets small when you can 
(like 10% of all data). The more training data, the better. 

But these are all arbitrary guidelines. No one can tell you how to hyper-optimize. 
You need to try optimizing a dataset for yourself to find your own methodology and tricks. 
The [dp GitHub repository](https://github.com/nicholas-leonard/dp/) 
also provides a [wiki](https://github.com/nicholas-leonard/dp/wiki/Hyperparameter-Optimization) 
that can be used to share hyper-parameter configurations
as well as corresponding performance metrics and observations. 
It is easier to hyper-optimize as a team than alone (everyone has a piece of the puzzle).
