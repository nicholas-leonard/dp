imagenetloader.torch
====================

some old code that i wrote, might be useful to others.

If your dataset is of the form:

* dogs/[image files of dogs]
* cats/[image files of cats]
* .
* .
* .

then this loader loads the dataset nicely, and you have class-balanced sampling, a test iterator and other useful things.

I also used it to load imagenet (both the 1.2million set and the 14 million full imagenet).

