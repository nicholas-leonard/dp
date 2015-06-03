# You can Observe a lot by just watching #
The following Observers are available:

  * [Observer](#dp.Observer) : abstract class;
    * [ErrorMinima](#dp.ErrorMinima) : monitors an error variable to keep track of its minima.
      * [EarlyStopper](#dp.EarlyStopper) : upon reaching minima, saves [Experiment](experiment.md#dp.Experiment) and notifies [Subscribers](mediator.md#dp.Subscriber);
   * [AdaptiveDecay](#dp.AdaptiveDecay) : decays learning rate when new minima on validation error isn't found;
   * [Logger](#dp.Logger) : abstract logging class (prints reports to cmdline);
     * [FileLogger](#dp.FileLogger) : basic file-based report logging;
   * [CompositeObserver](#dp.CompositeObserver) : a composite of observers;

<a name="dp.Observer"/>
[]()
## Observer ##
An object that is called when events occur. Based on the Listen-Notify design pattern. 
Uses a mediator to publish/subscribe to channels.
Observers cannot publish reports (for now). The reason for this is 
that a report is required of observers to do their job, thus making
it impossible for them to participate to report generation.
The only possibility would be to allow for Observers to modify the 
received report, but then the ordering of these modifications would
be undefined, unless they make use of Mediator priorities.

<a name='dp.Observer.__init'/>
[]()
### dp.Observer(channels, callbacks) ###
Constructs an Observer. Arguments `channels` and `callbacks` can 
be strings or tables thereof. The first argument specifies the name of 
a [Channel](mediator.md#dp.Channel) to which the corresponding `callback` method of the Observer 
will be registered via a [Subscriber](mediator.md#dp.Subscriber) through the 
[Mediator](mediator.md#dp.Mediator).
If no `callbacks` are provided, these default to taking on the same values 
as `channels`.

<a name='dp.ErrorMinima'/>
[]()
## ErrorMinima ##
Monitors an error variable to keep track of its minima. Optionally notifies
[Subscribers](mediator.md#dp.Subscriber) to the `errorMinima` of the current state 
of the error minima.

The monitored error variable can be obtained from an experiment 
report or a [Mediator](mediator.md#dp.Mediator) [Channel](mediator.md#dp.Mediator). 
In the case of the report, the variable must be 
located at a leaf of the report tree specified by a sequence of keys (branches) : `error_report`. 
In this case, the object subscribes to `doneEpoch` channel in order to receive the reports
at the end of each epoch.

<a name='dp.ErrorMinima.__init'></a>
### dp.ErrorMinima{...} ###

Constructs an ErrorMinima Observer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Observer](#dp.Observer.__init) also apply.
 
  * `start_epoch` is a number identifying the minimum epoch after which [Experiments](experiment.md#dp.Experiment) can begin to be persisted to disk. Useful to reduce I/O at beginning of training. Defaults to 5;
  * `error_report` is a table specifying a sequence of keys to access variable from epoch [reports](experiment.md#dp.Experiment.repot). Default is `{'validator', 'loss', 'avgError'}` (for cross-validation), unless of course an `error_channel` is specified.
  * `error_channel` is a string or table specifying the [Channel](mediator.md#dp.Channel) to subscribe to for early stopping. Should return an error value and the last experiment report. Over-rides `error_report` when specified.
  * `maximize` is a boolean having a default value of false. When true, the error channel or report is negated. This is useful when the channel returns a measure like accuracy that should be maximized, instead of an error that should be minimized.
  * `notify` is a boolean. When true (the default), notifies listeners ([Subscribers](mediator.md#dp.Subscriber)) when a new minima is found.
 
<a name="dp.EarlyStopper"></a>
## EarlyStopper ##
An [ErrorMinima](#dp.ErrorMinima) instance that saves the version of 
the `subject` with the lowest error and terminates 
the experiment when no new minima is found for `max_epochs`.
Should only be called on Experiment, Propagator or Model subjects.
Error can be obtained from experiment report or mediator Channel. 
If obtained from experiment report via error_report, subscribes to doneEpoch channel.

### dp.EarlyStopper{...} ###

Constructs an EarlyStopper Observer. Arguments should be specified as key-value pairs.
Other then the following arguments, those specified in [ErrorMinima](#dp.ErrorMinima.__init) also apply.
 
  * `save_strategy` is an object with a `save(subject)` method for persisting the subject to disk. Defaults to `dp.SaveToFile()`
  * `max_epochs` specifies the maximum number of epochs to consider after a minima has been found. After that, a terminate signal is published to the mediator. Defaults to 30.
  * `max_error` is the maximum value for which the experiment should be stopped after `min_epoch` epochs. If `maximize=true` this is minimum value. A value of 0 (the default) means that it is ignored.
  * `min_epoch` is a number having a default value of 1000000. See `max_value` for a description.

<a name="dp.AdaptiveDecay"></a>
## AdaptiveDecay ##

An Observer that decays learning rate by `decay_factor` when validation error doesn't reach a new 
minima for `max_wait` epochs. This object should observe in conjuction with an 
[ErrorMinima](#dp.ErrorMinima) instance, such as [EarlyStopper](#dp.EarlyStopper).

As example, suppose the object is initialized with `decay_factor=0.5` 
and `max_wait=1`, while the subject is initialized with `learning_rate=10`. 
If the sequence of errors is 
```lua
10, 9, 8, 8, 8, 8, 9, 7, 7, 8
```
then the corresponding sequence of learning rates given these errors would be
```lua
10, 10, 10, 10, 1, 1, 0.1, 0.1, 0.1, 0.01
```
 
<a name="dp.AdaptiveDecay.__init"></a>
### dp.AdaptiveDecay{...} ###

Constructs an AdaptiveDecay Observer. Arguments should be specified as key-value pairs.
Other then the following arguments, those specified in [Observer](#dp.Observer.__init) also apply.
 
  * `max_wait` specifies the maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by `decay_factor`. Defaults to 2.
  * `decay_factor` specifies the factor by which learning rate `lr` is decayed as per : `lr = lr*decay_factor`.

<a name="dp.Logger"/>
[]()
## Logger ##

<a name="dp.FileLogger"/>
[]()
## FileLogger ##

<a name="dp.CompositeObserver"/>
[]()
## CompositeObserver ##
