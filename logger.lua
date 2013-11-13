
------------------------------------------------------------------------
--[[ Logger ]]--
-- Responsible for keeping track of the experiment as it progresses
-- for later analysis.
------------------------------------------------------------------------
--[[ Discussion ]]--
--[[
The state of the experiment could be saved in an external object.
Which could be referenced by all interested objects, including 
Experiment, Propagators, Sampler, Logger, etc. It could also take the 
form of a table of keys and values which any object could modify. The 
advantage of this approach is that new keys can be added easily. With 
the current model involving passing around the experiment and making 
it accessible to everyone, we require object-oriented methods be 
pre-coded. Actually, we don't. Any object accessible through 
experiment can be extended with new methods by, for example, observers.

The advantage of a state is that it can be easily saved/serialized. 
On the other hand, if the objects are indiscernible from their state, 
then they will be serialized as long with all objects they reference...
So in the case of a pointer to a dataset, the dataset would also be 
serialized, which isn't what one wants.

In pylearn2, the monitor acts as a kind of publish/subscribe mediator,
where all channels are printed every epoch, and updated/averaged 
every batch. Which also serves as a kind of logger.

The logger could be implemented as the pylearn2 monitor where objects 
push data to it. The problem with this is that you end up with lots of 
channels for which you do not necessarily care about. It doesn't hurt
to print them to screen every epoch, but its does hurt to save them 
all to a database all the time. This is an argument for a logger 
implemented as an observer that extracts only the data it requires.

A logger is just a bunch of namespaces (files, tables) and 
channels (keys and values that can be converted to string).

# Structured SQL

The results of the logger are later used for analysis. If these 
take the form of disparate files, all files hosting data necessary 
for an analysis must be loaded into memory, parsed, aggregated, etc.
This can be very slow, and if divided into many files, will require 
that these be divided logically such that it is known in which files
a required channel is located. Furthermore, if using different compute 
clusters, files will be distributed on different file systems.

In SQL, we would have our data indexed by experiment_name and 
experiment_epoch. Each file in the previous scheme would be analogous 
to a table, where each column is its own channel. But this would make
channel names a hard constraint, i.e. a user couldn't easily add 
new channels unless a commensurate table is created or altered.

# Unstructured SQL

The main advantage of SQL is that it is very efficient at querying data,
as long as it is structured and indexed correctly. However, there is 
an approach to SQL that doesn't profit from this advantage so much, 
although it is much less structured. The idea is to have one table 
for every type of channel (type of key, type of value). And then 
each channel entry is indexed by experiment_name, experiment_epoch, 
channel_namespace, channel_name. 
Of course, optimizations can be applied to make this 
storage more efficient, such as hashing the experiment_name and 
channel_name to a unique key, etc. 

In any case, the rows of different tables need only be joined to 
create a view of the table. This is the approach of jobman's DD 
postgresql interface. The main advantage of course is that once the 
tables are created, no new ones really need to be. 
The main disadvantage is that Views must be created to facilitate 
analysis, which are a little bit of a pain to write, but we can imagine 
a lua script that would automate this process, and that storage costs 
are higher (more indexes, and more indexed columns, and more columns),
and that queries are slower (Views are constructed just-in-time).
Nevertheless, queries will be faster than looking through files.

# Analysis queries

What kind of analysis queries do we expect to require?
 * Learning curves comparing models.
   * Or more generally, Channel curves for comparing models.
 * Ordered tables of correlations between error and different hyper-parameters.
 * Bar-charts comparing 2-hyper-parameters and error, etc.
Each of these queries would be constrained to a particular set of 
experiments. And the experiment hyper-parameters would also be stored 
for later analysis.

# Lua NoSQL

The quickest way to implement all of this would be to serialize and 
store hyper-parameters states (used to initialize experiment) in 
a table, and store epoch states in another. Analysis queries are 
implemented as modules that can be accessed from the torch shell, such 
that data is unserialized and structured in memory the first time it is 
queried from the shell. 

In any case, storing hyper-parameter states requires an experiment 
factory, which takes the state for its constructor and returns a 
fully initialized experiment ready for optimization. The factory itself
would be called with a probability distribution over hyper-parameters
which would be used to sample experiment states. 

The database would be simple to implement, storing bytea arrays for 
the different states. Experiments would be grouped/tagged to faciliate
querying from the analysis module. 

The main disadvantage is the security risk of storing lua-code for 
later execution, but this can be aleviated by enforcing read-only access
to the database during analysis, etc. Another disadvantage is that the 
module will need to implement a kind of client-side caching to speedup
analysis. Yet another disadvantage is that we cannot use the facilities
of SQL to speedup analysis.

As for the epoch states, these would need to be their own separate 
objects for easy serialization. Whether the logger is updated with 
information from the experiments, or it gets the information itself, 
is irrelevant since in either case a state can be created. 

Advantages of this approach:
 * Simplicity of the database schema: 
  * No need to update it for every new object created. 
  * No need for complex dependencies
 * Quick to implement
  * Just need to implement logger as serializable state
  * Same for experiment constructor
 * Flexible queries 
  * Can use underscore with task-tailored funciton to query experiments.
  * Can use the full power of a programming language like lua
  * No class/table constraints
 * Lower storage cost compared to unstructured SQL approach (smaller/less indexes)
  
Disadvantages:
 * Can't use SQL to its full potential (for querying, indexing, etc)
  * Can still use it for building message queues, experiment groups, ...
 * Security risk (code injection)
 * Need to implement client-side caching/query analysis (i.e. set logic)
 
TODO
 * Read up on Memento design patterns.
 * Test serialization of our objects.
 * Configuration (Experiment Structured-Constructor)
 * Epoch (Monitoring information at one epoch)
 * Snapshot (model and experiment parameters required to restart experiment)
 * Analysis Module (queries, caching, etc)
 * Simple database schema
 * Experiment should be initializable from Snapshot
]]--

