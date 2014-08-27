# You can Observe a lot by just watching #
The following Observers are available:
 * [Observer](#dp.Observer) : abstract class;
  * [EarlyStopper](#dp.EarlyStopper) : upon reaching minima, saves [Experiment](experiment.md#dp.Experiment) and notifies [Subscribers](mediator.md#dp.Subscriber);
   * [PGEarlyStopper](#dp.PGEarlyStopper) : the PostgreSQL version of the EarlyStopper;
  * [LearningRateSchedule](#dp.LearningRateSchedule) : modifies learning rate using a schedule;
  * [Logger](#dp.Logger) : abstract logging class (prints reports to cmdline);
   * [FileLogger](#dp.FileLogger) : basic file-based report logging;
   * [PGLogger](#dp.PGLogger) : a PostgreSQL logger which saves hyper-parameters, reports and minima;
  * [CompositeObserver](#dp.CompositeObserver) : a composite of observers;

<a name="dp.Observer"/>
## Observer ##
An object that is called when events occur. Based on the Listen-Notify design pattern. 
Uses a mediator to publish/subscribe to channels.
Observers cannot publish reports (for now). The reason for this is 
that a report is required of observers to do their job, thus making
it impossible for them to participate to report generation.
The only possibility would be to allow for Observers to modify the 
received report, but then the ordering of these modifications would
be undefined, unless they make use of Mediator priorities.

<a name="dp.EarlyStopper"/>
## EarlyStopper ##
Saves version of the subject with the lowest error and terminates 
the experiment when no new minima is found for `max_epochs`.
Should only be called on Experiment, Propagator or Model subjects.
Error can be obtained from experiment report or mediator Channel. 
If obtained from experiment report via error_report, subscribes to doneEpoch channel.

### dp.EarlyStopper{} ###
EarlyStopper constructor. Arguments should be specified as key-value pairs:
 * `start_epoch` is a number identifying the minimum epoch after which [Experiments](experiment.md#dp.Experiment) can begin to be persisted to disk. Useful to reduce I/O at beginning of training. Defaults to 5;
 * `error_report` is a table specifying a sequence of keys to access error from epoch [reports](experiment.md#dp.Experiment.repot). Default is `{'validator', 'loss', 'avgError'}` (for cross-validation), unless of course an `error_channel` is specified.
 * `error_channel` is a string or table specifying the [Channel](mediator.md#dp.Channel) to subscribe to for early stopping. Should return an error which the Model should be minimized, and the report of the experiment. Over-rides `error_report` when specified.
 * `maximize` is a boolean having a default value of false. When true the error channel or report is negated. This is useful when the channel returns a measure like accuracy that should be maximized, instead of an error that should be minimized.
 * `save_strategy` is an object with a `save(subject)` method for persisting the subject to disk. Defaults to `dp.SaveToFile()`
 * `max_epochs` specifies the maximum number of epochs to consider after a minima has been found. After that, a terminate signal is published to the mediator. Defaults to 30.
 * `max_error` is the maximum value for which the experiment should be stopped after `min_epoch` epochs. If `maximize=true` this is minimum value. A value of 0 (the default) means that it is ignored.
 * `min_epoch` is a number having a default value of 1000000. See `max_value` for a description.

<a name="dp.PGEarlyStopper"/>
## PGEarlyStopper ##

<a name="dp.LearningRateSchedule"/>
## LearningRateSchedule ##

<a name="dp.Logger"/>
## Logger ##

<a name="dp.FileLogger"/>
## FileLogger ##

<a name="dp.PGLogger"/>
## PGLogger ##

<a name="dp.CompositeObserver"/>
## CompositeObserver ##
