<a name="NeuralNetworkTutorial"/>
[]()
# Neural Network Tutorial #

We begin with a simple [neural network example](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork.lua). 
The first line loads the __dp__ package, whose first matter of business is to load its dependencies 
(see [init.lua](https://github.com/nicholas-leonard/dp/blob/master/init.lua)):

```lua
require 'dp'
```

Note : package [Moses](https://github.com/Yonaba/Moses) is imported as `_`. 
So `_` shouldn't be used for dummy variables. 
Instead use the much more annoying `__`, or whatnot. 

## Command-line Arguments ##

Lets define some command-line arguments. 
These will be stored into table `opt`, which will be printed when the script is launched.
Command-line arguments make it easy to control the experiment and 
try out different hyper-parameters without needing to modify any code.

```lua
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end
```

## Preprocess ##

The `--standardize` and `--zca` cmd-line arguments can be toggled on
to perform some preprocessing on the data. 
```lua
--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end
```

A very common and easy preprocessing technique is to [Standardize](preprocess.md#dp.Standardize) 
the datasource, which subtracts the mean and divides 
by the standard deviation. Both statistics (mean and standard deviation) are 
measured on the `train` set only. This is a common pattern when preprocessing data. 
When statistics need to be measured across different examples 
(as in [ZCA](preprocess.md#dp.ZCA) and [LecunLCN](preprocess.md#dp.LeCunLCN) preprocesses), 
we fit the preprocessor on the `train` set and apply it to all sets (`train`, `valid` and `test`). 
However, some preprocesses require that statistics be measured
only on each example, as is the case for global constrast normalization ([GCN]](preprocess.md#dp.GCN)),
such that there is no fitting. 

## DataSource ##

We intend to build and train a neural network so we need some data, 
which we encapsulate in a [DataSource](data.md#dp.DataSource)
object. __dp__ provides the option of training on different datasets, 
notably [MNIST](data.md#dp.Mnist), [NotMNIST](data.md#dp.NotMnist), 
[CIFAR-10](data.md#dp.Cifar10) or [CIFAR-100](data.md#dp.Cifar100). 
The default for this script is the archetypal MNIST (don't leave home without it).
However, you can use the `--dataset` argument to specify a different image classification
dataset.

```lua
--[[data]]--

if opt.dataset == 'Mnist' then
   ds = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   ds = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   ds = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   ds = dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end
```

A DataSource contains up to three [DataSets](data.md#dp.DataSet): 
`train`, `valid` and `test`. The first is for training the model. 
The second is used for [early-stopping](observer.md#dp.EarlyStopper) and cross-validation.
The third is used for publishing papers and comparing results across different models.

## Model of Modules ##

Ok so we have a DataSource, now we need a model. Let's build a 
multi-layer perceptron (MLP) with one or more parameterized non-linear layers
(note that in the case of hidden layers being ommitted (`--hiddenSize '{}'`), 
the model is just a linear classifier):

```lua
--[[Model]]--

model = nn.Sequential()
model:add(nn.Convert(ds:ioShapes(), 'bf')) -- to batchSize x nFeature (also type converts)

-- hidden layers
inputSize = ds:featureSize()
for i,hiddenSize in ipairs(opt.hiddenSize) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   if opt.batchNorm then
      model:add(nn.BatchNormalization(hiddenSize))
   end
   model:add(nn.Tanh())
   if opt.dropout then
      model:add(nn.Dropout())
   end
   inputSize = hiddenSize
end

-- output layer
model:add(nn.Linear(inputSize, #(ds:classes())))
model:add(nn.LogSoftMax())
```

Output and hidden layers are defined using a [Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear),
which contains the parameters that will be learned, followed by a non-linear transfer function like 
[Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh) (for the hidden neurons) 
and [LogSoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.LogSoftMax) (for the output layer).
The latter might seem odd (why not use [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.SoftMax) instead?), 
but the [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion) only works 
with LogSoftMax (or with SoftMax + [Log](https://github.com/torch/nn/blob/master/Log.lua)).

The Linear modules are constructed using 2 arguments, 
`inputSize` (number of input units) and `outputSize` (number of output units).
For the first layer, the `inputSize` is the number of features in the input image. 
In our case, that is `1x28x28=784`, which is what `ds:featureSize()` will return.

Now for the odd looking [nn.Convert](https://github.com/nicholas-leonard/dpnn#nn.Convert) Module. It has two purposes. First, 
whatever type of Tensor received, it will output the type of Tensor used by the Module.
Second, it can convert from different Tensor `shapes`. The `shape` of a typical image is 
*bchw*, short for  *batch*, *color/channel*, *height* and *width*. Modules like 
[SpatialConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution) 
and [SpatialMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialMaxPooling) 
expect this type of input. Our MLP, on the other hand, expects an input of shape *bf*, short for *batch*, *feature*. 
Its a pretty simple conversion actually, all you need to do is flatten the *chw* 
dimensions to a single *f* dimension (in this case, of size 784). 

For those not familiar with the nn package, all the `nn.*` in the above snippet of code 
are [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) subclasses. 
This is true even for the [Sequential](https://github.com/torch/nn/blob/master/containers.md#nn.Sequential).
Although the latter is special. It is a [Container](https://github.com/torch/nn/blob/master/containers.md#nn.Container) 
of other Modules, i.e. a [composite](http://en.wikipedia.org/wiki/Composite_pattern).

## Propagator ##

Next we initialize some [Propagators](propagator.md#dp.Propagator). 
Each such Propagator will propagate examples from a different [DataSet](data.md#dp.DataSet).
[Samplers](data.md#dp.Sampler) iterate over DataSets to 
generate [Batches](data.md#dp.Batch) of examples (inputs and targets) to propagate through the `model`:
```lua
--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
   acc_update = opt.accUpdate,
   loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
   epoch_callback = function(model, report) -- called every epoch
      -- learning rate decay
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

```

For this example, we use an [Optimizer](propagator.md#dp.Optimizer) for the training DataSet,
and two [Evaluators](propagator.md#dp.Evaluator), one for cross-validation 
and another for testing. Now lets explore the different constructor arguments.

### `sampler` ###

The Evaluators use a simple Sampler which iterates sequentially through the DataSet. 
On the other hand, the Optimizer uses a [ShuffleSampler](data.md#dp.SuffleSampler). 
This Sampler shuffles the (indices of a) DataSet before each pass over all examples in a DataSet. 
This shuffling is useful for training since the model must learn from varying sequences of batches through the DataSet.
Which makes the training algorithm more stochastic (subject to the constraint that each example is presented once and only once per epoch).

### `loss` ###

Each Propagator can also specify a `loss` for training or evaluation. This argument is 
only mandatory for the Optimizer, as it is required for [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
If you have previously used the [nn](https://github.com/torch/nn/blob/master/README.md) package, 
there is nothing new here. The `loss` is a [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion). 
Each example has a single target class and our Model output is LogSoftMax so 
we use a [ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion).
The criterion is wrapped in [ModuleCriterion](https://github.com/nicholas-leonard/dpnn/blob/master/README.md#nn.ModuleCriterion),
which is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) that allows us to pass 
each `input` and `target` through a module before it is passed on to the decorated `criterion`. 
In our case, we want to make sure each `target` gets converted to the type of the `loss`. 

### `feedback` ###

The `feedback` parameter is used to provide us with, you guessed it, feedback (like performance measures and
statistics after each epoch). We use [Confusion](feedback.md#dp.Confusion), which is a wrapper 
for the [optim](https://github.com/torch/optim/blob/master/README.md) package's 
[ConfusionMatrix](https://github.com/torch/optim/blob/master/ConfusionMatrix.lua).
While our Loss measures the Negative Log-Likelihood (NLL) of the `model` 
on different datasets, our [Feedback](feedback.md#feedback) 
measures classification accuracy (which is what we will use for 
early-stopping and comparing our model to the state-of-the-art).

### `callback` ###

Since the [Optimizer](propagator.md#dp.Optimizer) is used to train the `model` on a DataSet, 
we need to specify a `callback` function that will be called after successive `forward/backward` calls.
Among other things, the callback should either 
[updateParameters](https://github.com/nicholas-leonard/dpnn#nn.Module.updateParameters) 
or [accUpdateGradParameters](https://github.com/nicholas-leonard/dpnn#nn.Module.accUpdateGradParameters). 
Depending on what is specified in the command-line, it can also be used to 
[updateGradParameters](https://github.com/nicholas-leonard/dpnn#nn.Module.updateGradParameters) 
(commonly known as momentum learning). You can also choose to regularize it with 
[weightDecay](https://github.com/nicholas-leonard/dpnn#nn.Module.weightDecay) or 
[maxParamNorm](https://github.com/nicholas-leonard/dpnn#nn.Module.maxParamNorm), (personally, I prefer the latter to the former). 
In any case, the `callback` is a function that you can define to fit your needs.

### `epoch_callback` ###

While the `callback` argument is called every batch, the `epoch_callback` is called between epochs.
This is useful for decaying hyper-parameters such as the learning rate, which is what we do in this example.
Since the learning rate is the most important hyper-parameter, it is a good idea to try 
different learning rate decay schedules during hyper-optimization. 

The `--lrDecay 'linear'` decay is the easiest to use (the default cmd-line argument). 
It involves specifying a starting learning rate `--learningRate`, a minimum learning rate `--minLR` and the 
epoch at which that minimum will be reached : `--saturateEpoch`. 

The `--lrDecay 'adaptive'` uses an exponentially decaying learning rate. 
By default this mode only decays the learning rate when a minima hasn't been found for `--maxWait` epochs. 
But by using `--maxWait -1`, we can decay the learning rate 
every epoch with the following rule : `lr = lr*decayFactor`. 
This will decay the learning rate much faster than a linear decay. 

Another option (`--lrDecay `schedule`) is to specify the learning rate 
`--schedule` manually by specifying a table mapping learning rates to epochs like '{[200] = 0.01, [300] = 0.001}', 
which will decay the learning rate to the given value at the given epoch.

Of course, because this argument is just another callback function, you can use it however you please 
by coding your own function.

### `acc_update` ###

When set to true, the gradients w.r.t. parameters (a.k.a. `gradParameters`) 
are accumulated directly into the parameters (a.k.a. `parameters`) to produce
an update after the `forward` and `backward` pass. 
In other words, for `acc_update=true`, the sequence for propagating a batch is essentially:

 1. `updateOutput`
 2. `updateGradInput`
 3. `accUpdateGradParameters`
 
Instead of the more common: 

 1. `updateOutput`
 2. `updateGradInput`
 3. `accGradParameters`
 4. `updateParameters`
 
This means that no `gradParameters` are actually used internally. The default value if false.
Some methods do not work with `acc_update` as they require the the `gradParameters` tensors be populated before being 
added to the `parameters`. This is the case for `updateGradParameters` (momentum learning) and `weightDecay`.

### `progress` ###

Finally, we allow for the Optimizer's `progress` bar to be switched on so that we 
can monitor training progress. 

## Experiment ##

Now its time to put this all together to form an [Experiment](experiment.md):
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

The Experiment can be initialized using a list of [Observers](observer.md#dp.Observer). The 
order is not important. Observers listen to mediator [Channels](mediator.md#dp.Channel). The Mediator 
calls them back when certain events occur. In particular, they may listen to the _doneEpoch_
Channel to receive a report from the Experiment after each epoch. A report is nothing more than 
a bunch of nested tables matching the object structure of the experiment. 
After each epoch, the component objects of the Experiment (except Observers) 
can each submit a report to its composite parent thereby forming a tree of reports. The Observers can analyse 
these and modify the components which they are assigned to (in this case, Experiment). 
Observers may be attached to Experiments, Propagators, Visitors, etc. 

#### FileLogger ####

Here we use a simple [FileLogger](observer.md#dp.FileLogger) which will 
store serialized reports in a simple text file for later use. Each experiment has a unique ID which is 
included in the corresponding reports, thus allowing the FileLogger to name its file appropriately.

#### EarlyStopper ####

The [EarlyStopper](observer.md#dp.EarlyStopper) is used for stopping the Experiment 
when the error has not decreased, or accuracy has not been maximized. 
It also saves to disk the best version of the Experiment when it finds a new one. 
It is initialized with a channel to `maximize` or minimize (the default is to minimize). In this case, we intend 
to early-stop the experiment on a field of the report, in particular the _accuracy_ field of the 
_confusion_ table of the _feedback_ table of the `validator`. 
This `{'validator','feedback','confusion','accuracy'}` happens to measure the accuracy of the Model on the 
validation DataSet after each training epoch. So by early-stopping on this measure, we hope to find a 
Model that [generalizes](http://en.wikipedia.org/wiki/Generalization_error) well. 
The parameter `max_epochs` indicates how many consecutive epochs of training can occur 
without finding a new best model before the experiment is signaled to stop 
by the _doneExperiment_ Mediator Channel.

## Running the Experiment ##

Once we have initialized the experiment, we need only run it on the `datasource` to begin training.

```lua
xp:run(ds)
```

We don't initialize the Experiment with the DataSource so that we may easily 
save it to disk, thereby keeping this snapshot separate from its data 
(which shouldn't be modified by the experiment).

Let's run the [script](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork.lua) 
from the cmd-line (with default arguments):

```bash
nicholas@xps:~/projects/dp$ th examples/neuralnetwork.lua 
```
First it prints the command-line arguments stored in `opt`:
```bash
{
   batchNorm : false
   batchSize : 32
   cuda : false
   dataset : "Mnist"
   dropout : false
   hiddenSize : {200,200}
   learningRate : 0.1
   lecunlcn : false
   maxEpoch : 100
   maxOutNorm : 1
   maxTries : 30
   momentum : 0
   progress : false
   schedule : {[200]=0.01,[400]=0.001}
   silent : false
   standardize : false
   useDevice : 1
   zca : false
}	
```
After that it prints the model.
```bash
Model :	
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> output]
  (1): nn.Convert
  (2): nn.Linear(784 -> 200)
  (3): nn.Tanh
  (4): nn.Linear(200 -> 200)
  (5): nn.Tanh
  (6): nn.Linear(200 -> 10)
  (7): nn.LogSoftMax
}
```
The `FileLogger` then prints where the epoch logs will be saved. This
can be controlled with the `$TORCH_DATA_PATH` or `$DEEP_SAVE_PATH` 
environment variables. It defaults to `$HOME/save`.
```bash
FileLogger: log will be written to /home/nicholas/save/xps:1432747515:1/log	
```
Finally, we get to the fun part : the actual training. Every epoch, 
some performance data gets printed to `stdout`:
```bash
==> epoch # 1 for optimizer :	
==> example speed = 4508.3691689025 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.012714211946021	
xps:1432747515:1:optimizer:confusion accuracy = 0.8877	
xps:1432747515:1:validator:confusion accuracy = 0.9211	
xps:1432747515:1:tester:confusion accuracy = 0.9292	
==> epoch # 2 for optimizer :	
==> example speed = 4526.7213369494 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0072034133582363	
xps:1432747515:1:optimizer:confusion accuracy = 0.93302	
xps:1432747515:1:validator:confusion accuracy = 0.9405	
xps:1432747515:1:tester:confusion accuracy = 0.9428	
==> epoch # 3 for optimizer :	
==> example speed = 4486.8207535058 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0056732489919492	
xps:1432747515:1:optimizer:confusion accuracy = 0.94704	
xps:1432747515:1:validator:confusion accuracy = 0.9512	
xps:1432747515:1:tester:confusion accuracy = 0.9518	
==> epoch # 4 for optimizer :	
==> example speed = 4524.4831336064 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0047361240094285	
xps:1432747515:1:optimizer:confusion accuracy = 0.95672	
xps:1432747515:1:validator:confusion accuracy = 0.9565	
xps:1432747515:1:tester:confusion accuracy = 0.9584	
==> epoch # 5 for optimizer :	
==> example speed = 4527.7260154406 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0041567858616232	
xps:1432747515:1:optimizer:confusion accuracy = 0.96188	
xps:1432747515:1:validator:confusion accuracy = 0.9603	
xps:1432747515:1:tester:confusion accuracy = 0.9613	
SaveToFile: saving to /home/nicholas/save/xps:1432747515:1.dat	
==> epoch # 6 for optimizer :	
==> example speed = 4519.2735741475 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0037086909102431	
xps:1432747515:1:optimizer:confusion accuracy = 0.9665	
xps:1432747515:1:validator:confusion accuracy = 0.9602	
xps:1432747515:1:tester:confusion accuracy = 0.9629	
==> epoch # 7 for optimizer :	
==> example speed = 4528.1356378239 examples/s	
xps:1432747515:1:optimizer:loss avgErr 0.0033203622647625	
xps:1432747515:1:optimizer:confusion accuracy = 0.97062	
xps:1432747515:1:validator:confusion accuracy = 0.966	
xps:1432747515:1:tester:confusion accuracy = 0.9665	
SaveToFile: saving to /home/nicholas/save/xps:1432747515:1.dat	
```
After 5 epochs, the experiment starts early-stopping by saving to disk
the version of the model with the lowest `xps:1432747515:1:validator:confusion accuracy`.
The first part of that string (`xps:1432747515:1`) is the unique id of the experiment.
It concatenates the hostname of the computer (`xps` in this case) and a time-stamp.

## Loading the saved Experiment ##

The experiment is saved at `/home/nicholas/save/xps:1432747515:1.dat`. You can 
load it and access the `model` with  :

```lua
require 'dp'
require 'cuda' -- if you used cmd-line argument --cuda
require 'optim'

xp = torch.load("/home/nicholas/save/xps:1432747515:1.dat")
model = xp:model()
print(torch.type(model))
nn.Serial
```

For efficiency, the `model` here is decorated with a [nn.Serial](https://github.com/nicholas-leonard/dpnn#nn.Serial).
You can access the `model` you passed to the experiment by adding :

```lua
model = model.module
print(torch.type(model))
nn.Sequential
```

## Hyper-optimization ##

Hyper-optimization is the hardest part of deep learning. 
In many ways, it can feel more like an art than a science. 

A got this question from a dp user : 
> If I am attempting to train a NN with a custom dataset - how do I optimize the parameters ? 
> I mean, if the NN takes about 3-5 days to train completely.. 
>  How do you do small experiments / tuning ( learning rate, nodes in each layer )
> - so that you dont spend 5 days with each - 
> Basically what is the quickest way to find the best parameters ?

This is what I answered :
> In my opinion it is a mix of experience and computing resources. 
> The first comes from playing around with different combinations of datasets and models. 
> Eventually, you just know what kind of regime work well. 
> As for the second, it lets you try different hyper-parameters at the same time. 
> If you have access to 4-8 GPUs, that is 4-8 experiments to try at once in parallel. 
> You can also quickly see that some of those experiments are mush slower to converge than the others.
> In which case you can kill them and try something similar to the ones that work. 
> Also, the most important hyper-parameter is the learning rate. 
> Find the highest value that doesn't make the training diverge. 
> That way your experiments will converge faster.
> You can also decay it towards the end to get a little accuracy boost (not always).


I could have said more. For one, I recommend regularizing with `maxParamNorm`. 
A `max_out_norm` around 2 is usually a good starting point, continuing with 1, 10, 
and only try -1 when out of ideas. 
Second, you can vary the epoch sizes to better divide processing time between evaluation and training. 
It's often best to keep the evaluation sets small when you can 
(like less than 10% of all data). Also, the more training data, the better. 

But these are all arbitrary guidelines. No one can tell you how to hyper-optimize. 
You need to try optimizing a dataset for yourself to find your own methodology and tricks. 
The [dp GitHub repository](https://github.com/nicholas-leonard/dp/) 
also provides a [wiki](https://github.com/nicholas-leonard/dp/wiki/Hyperparameter-Optimization) 
that can be used to share hyper-parameter configurations
as well as corresponding performance metrics and observations. 

Finally, it is easier to hyper-optimize as a team than alone. 
Everyone has a piece of the puzzle. For example, not too long ago I was 
reminded the importance of a good spreadsheet for keeping track of what 
experiments were tried. This guy would have a column for each hyper-parameter 
in the cmd-line arguments, columns for error/accuracy and observations, and a row for each experiment.
For me, I was too lazy to make this nice spreadsheet on my own, I just used paper and pen...
But then I noticed how easy it was for him to find the best hyper-parameter configurations.
Anyway, the point is, I learned from this guy who had this part of the puzzle all figured out.
So keep your eyes peeled for such lessons in the art of hyper-optimization.
