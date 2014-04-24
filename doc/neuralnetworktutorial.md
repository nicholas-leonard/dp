<a name="NeuralNetworkTutorial"/>
# Neural Network Tutorial #
We begin with a simple [neural network example](../examples/neuralnetwork_tutorial.lua). The first line loads 
the __dp__ package, whose first matter of business is to load its dependencies (see [init.lua](../init.lua)):
```lua
require 'dp'
```
Note : package [Moses](https://github.com/Yonaba/Moses/blob/master/docs/moses.md) is imported as `_`. So `_` shouldn't be used for dummy variables, instead 
use the much more annoying `__`, or whatnot. 

Lets define some hyper-parameters and store them in a table. We will need them later:
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
which we encapsulate in a [DataSource](../data/datasource.lua)
object. __dp__ provides the option of training on different datasets, 
notably [MNIST](../data/mnist.lua), [NotMNIST](../data/notmnist.lua), 
[CIFAR-10](../data/cifar10) or [CIFAR-100](../data/cifar100.lua), but for this
tutorial we will be using the archtypical MNIST (don't leave home without it):
```lua
--[[data]]--
datasource = dp.Mnist{input_preprocess = dp.Standardize()}
```
A `datasource` contains up to three [Datasets](../data/dataset.lua): 
`train`, `valid` and `test`. The first if for training the model. 
The second is used for [early-stopping](observer/earlystopper.lua).
The third is used for publishing papers and comparing different models.
  
Although not really necessary, we [Standardize](../preprocess/standardize.lua) 
the datasource, which subtracts the mean and divides 
by the standard deviation. Both statistics (mean and standard deviation) are 
measured on the `train` set only. This is common pattern when preprocessing. 
When statistics need to be measured accross different examples 
(as in [ZCA](../preprocess/zca.lua) and [LecunLCN](../preprocess/lecunlcn.lua) preprocesses), 
we fit the preprocessor on the `train` set and apply it to all sets (`train`, `valid` and `test`). 
However, some preprocesses require that statistics be measured
only on each example (as in [global constrast normalization](../preprocess/gcn.lua)). 

## Model of Modules ##
Ok so we have a `datasource`, now we need a `model`. Lets build a 
multi-layer perceptron (MLP) with two parameterized non-linear layers:
```lua
--[[Model]]--
model = dp.Sequential{
   models = {
      dp.Neural{
         input_size = datasource:featureSize(), 
         output_size = opt.nHidden, 
         transfer = nn.Tanh()
      },
      dp.Neural{
         input_size = opt.nHidden, 
         output_size = #(datasource:classes()),
         transfer = nn.LogSoftMax()
      }
   }
}
```
Both layers are defined using [Neural](../model/neural.lua), which require an `input_size` 
(number of input neurons), an `output_size` (number of output neurons) and a 
[transfer function](https://github.com/torch/nn/blob/master/README.md#nn.transfer.dok).
We use the `datasource:featureSize()` and `datasource:classes()` methods to access the
number of input features and output classes, respectively. As for the number of 
hidden neurons, we use our `opt` table of hyper-parameters. The `transfer` functions 
used are the `nn.Tanh()` (for the hidden neurons) and `nn.LogSoftMax` (for the output neurons).
The latter might seem odd (why not use `nn.SoftMax` instead?), but the the `nn.ClassNLLCriterion` 
only works with `nn.LogSoftMax`. Besides, unlike `nn.SoftMax`, `nn.LogSoftMax` is implemented in 
[cutorch](https://github.com/torch/cutorch), which means it can run on CUDA-capable GPUs.

Both models initialize parameters using the default sparse initialization 
(see [Martens 2010](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf)). 
If you construct it with argument `sparse_init=false`, it will delegate parameter initialization to 
`nn.Linear`, which is what `Neural` uses internally for its parameters.

These two `Neural` models are combined to form an MLP using the [Sequential](../model/sequential.lua), 
which is not to be confused with (yet very similar to), 
[nn.Sequential](https://github.com/torch/nn/blob/master/README.md#sequential). It differs in that
it can be constructed from a list of [Models](../model/model.lua) instead of 
[nn.Modules](https://github.com/torch/nn/blob/master/README.md#nn.Modules). `Models` have extra 
methods, allowing them to be `accept` [Visitors](visitor/visitor.lua), to communicate with 
other components through a [Mediator](../mediator.lua), or `setup` with variables after initialization.
`Model` instances also differ from `nn.Module` in their ability to `forward` and `backward` 
complex `states`.  

## Propagator ##
Next we initialize some [Propagators](../propagator/propagator.lua). 
Each such `propagator` will propagate examples from a different `dataset`.
[Samplers](../data/sampler.lua) iterate over `datasets` to 
generate batches of examples (inputs and targets) to propagated through 
the `model`:
```lua
--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor = opt.momentum},
      dp.Learn{learning_rate = opt.learningRate},
      dp.MaxNorm{max_out_norm = opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{}
}
test = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),
   sampler = dp.Sampler{}
}
```
For this example, we use an [Optimizer](../propagator/optimizer.lua) for the training set,
and two [Evaluators](../propagator/evaluator.lua), one for cross-validation 
and another for testing. 

### Sampler ###
The evaluators use the a simple `sampler` which 
iterates sequentially through the dataset. On the other hand, the optimizer 
uses a `ShuffleSampler` to iterate through the dataset. This `sampler` 
shuffles the `dataset` before each epoch (an epoch is a complete iteration 
over a `dataset`). This shuffling is useful for training since the model 
must learn from varying sequences of batches at each epoch, which makes the
training algorithm more stochastic.

### Loss ###
Each propagator must also specify a `loss` for training or evaluation.
If you have previously used the `nn` package, there is nothing new here, 
a [Loss](loss/loss.lua) is simply an adapter of [nn.Criterion](https://github.com/torch/nn/blob/master/README.md#nn.Criterion). 
Each example has a single target class and our model output is `nn.LogSoftMax` so 
we use a [NLL](loss/nll.lua), which wraps a  [nn.ClassNLLCriterion](https://github.com/torch/nn/blob/master/README.md#nn.ClassNLLCriterion).

### Feedback ###
The `feedback` parameter is used to provide us with feedback like performance measures and
statistics after each epoch. We use [Confusion](../feedback/confusion.lua), which is a wrapper 
for the [optim](https://github.com/torch/optim/blob/master/README.md) package's 
[ConfusionMatrix](https://github.com/torch/optim/blob/master/ConfusionMatrix.lua).
While our `criterions` measure the Negative Log-Likelihood (NLL) of the model 
on different datasets, our `feedbacks` measure classification accuracy (which is what 
we will use for early-stopping and comparing our model to the state of the art).

### Visitor ###
Since the optimizer is used to train the model on a dataset, we all need to specify some 
visitors to update its parameters. We want to update the model by sequentially appling 
three visitors: 
 1. [Momentum](../visitor/momentum.lua) : updates parameter gradients using a factored mixture of current and previous gradients.
 2. [Learn](../visitor/learn.lua) : updates the parameters using the gradients and a learning rate.
 3. [MaxNorm](../visitor/maxnorm.lua) : updates output or input neuron weights (in this case, output) so that they have a norm less or equal to a specified value.

The only mandatory visitor is the second one, which does the actual parameter updates (learning). The first is the well known 
momentum. The last is the lesser known hard constraint on the norm of output or input neuron weights 
(see [Hinton 2012](http://arxiv.org/pdf/1207.0580v1.pdf)), which acts as a regularizer. You could also
replace it with a more classic regularizer like [WeightDecay](../visitor/weightdecay.lua), in which case you 
would have to put it before the `Learn` visitor.

Finally, we have the optimizer switch on its `progress` bar so we 
can monitor its progress during training. 

## Experiment ##
Now its time to put this all togetherto form an [Experiment](../propagator/experiment.lua):
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
The experiment can be initialized with a list of [Observers](../observer/observer.lua). The 
order is not important. Observers listen to mediator [Channels](mediator.lua). The mediator 
calls them back when certain events occur. In particular, they may listen to the `"doneEpoch"`
channel to receive a report from the experiment after each epoch. A report is nothing more than 
a hierarchy of tables. After each epoch, the components objects of the experiment (except `Observers`) 
can submit a report to its composite parent thereby forming a tree of reports. The observers can analyse 
these and modify the component which they are assigned to (in this case, experiment). 
Observers may be attached to experiments, propagators, visitors, etc. 

#### FileLogger ####
Here we use a simple [FileLogger](../observer/filelogger.lua) which will 
store serialized reports in a simple text file for later use. Each experiment has a unique ID which are 
included in reports, thus allowing the `FileLogger` to name its file appropriately. 

#### EarlyStopper ####
The [EarlyStopper](../observer/earlystopper.lua) is used for stopping the experiment when error has not decreased, or accuracy has not 
be maximized. It also saves onto disk the best version of the experiment when it finds a new one. 
It is initialized with a channel to `maximize` or minimize (default is to minimize). In this case we intend 
to early-stop the experiment on a field of the report, in particular the `accuracy` field of the 
`confusion` table of the `feedback` table of the `validator`. 
This `{'validator','feedback','confusion','accuracy'}` happens to measure the accuracy of the model on the 
validation `dataset` after each training epoch. So by early-stopping on this measure, we hope to find a 
model that generalizes well. The parameter `max_epochs` indicates how much consecutive 
epochs of training can occur without finding a new best model before the experiment is signaled to stop 
on the `"doneExperiment"` mediator channel.

## Running the Experiment ##

Once we have initialized the experiment, we need only run it on the `datasource` to begin training.
```lua
xp:run(datasource)
```
We don't initilize the experiment with the datasource so that we may easily 
save it onto disk, thereby keeping this snapshot separate from its data 
(which shouldn't be modified by the experiment).

Let's run the [script](../examples/neuralnetwork_tutorial.lua) from the cmd-line:
```
nicholas@xps:~/projects/dp$ th examples/neuralnetwork_tutorial.lua 
checking for file located at: 	/home/nicholas/data/mnist/mnist-th7.tgz	
checking for file located at: 	/home/nicholas/data/mnist/mnist-th7.tgz	
checking for file located at: 	/home/nicholas/data/mnist/mnist-th7.tgz	
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
