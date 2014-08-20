# You can Observe a lot by just watching #
The following Observers are available:
 * [Observer](#dp.Observer)
   * [EarlyStopper](#dp.EarlyStopper)
    * [PGEarlyStopper](#dp.PGEarlyStopper)
   * [LearningRateSchedule](#dp.LearningRateSchedule)
   * [Logger](#dp.Logger)
    * [FileLogger](#dp.FileLogger)
    * [PGLogger](#dp.PGLogger)
   * [CompositeObserver](#dp.CompositeObserver) 

<a name="dp.Observer"/>
## Observer ##
An object that is called when events occur.
Based on the Listen-Notify design pattern. 
Uses a mediator to publish/subscribe to channels.
Observers cannot publish reports (for now). The reason for this is 
that a report is required of observers to do their job, thus making
it impossible for them to participate to report generation.
The only possibility would be to allow for Observers to modify the 
received report, but then the ordering of these modifications would
be undefined, unless they make use of Mediator priorities.

<a name="dp.EarlyStopper"/>
## EarlyStopper ##

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
