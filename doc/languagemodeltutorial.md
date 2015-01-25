# Language Model Tutorial #

In this tutorial, we will explore the implementation of 
language models using dp and nn. Language models (LM) are a sub-field of 
statistical natural language processing (NLP). 
Many approaches and variations of the concept exist, but the basic idea is to predict the next word, 
given a context of previous words. 

## One Billion Words Benchmark ##
For our experiments, we use the one billion words benchmark used for 
measuring progress in language models (\cite{chelba2013one}).
The dataset contains one billion words. The task consists in using the 
previous $n$ words (the context) to predict the next word (the target word).
All sentences are shuffled such that only the words from the target's sentence 
can be used for prediction. The end of the sentence, identified 
by token "</S>", must also be predicted. To predict the first $n$ words of a sentence, 
padding is added. Using sentence "<S> Nicholas is writing</S>"
as an example, each input -> target word would have the following contexts of 3 words:

  * "</S> </S> <S>" -> "Nicholas" ;
  * "</S> <S> Nicholas" -> "is" ;
  * "<S> Nicholas is" -> "writing" ; and
  * "Nicholas is writing" -> "</S>".

The entire dataset is divided into 100 partitions of equal size,
99 of which are used for training. The remaining partition is further 
divided into 50 partitions, one of which is used for testing,
while the remaining 49 are reserved for cross-validation. 
All words with less then 3 occurrences in the training set are replaced 
with the "<UNK>" token. This is the same split used in (\cite{chelba2013one}).
The dataset is wrapped by the [BillionWords](data.md#dp.BillionWords) [DataSource](data.md#dp.DataSource).
The downloaded `billionwords.tar.gz` compressed tarball contains the following files:
 
  * `train_data.th7`, `train_small.th7` and `train_tiny.th7` training sets of different size (from fullest to smallest) ;
  * `valid_data.th7` the validation set ;
  * `test_data.th7` the test set ;  
  * `word_freq.th7` the frequencies of words (not really used) ;  
  * `word_tree1.th7`, `word_tree2.th7` and `word_tree3.th7` different hierarchies of words ; and
  * `word_map.th7` maps the word IDs (efficient integers) to the actual words (bulky strings). 
  
The training, validation and test set files contain serialized 2D Tensors.
Each such Tensor has 2 columns. First column is for storing start indices of sentences. 
Second column is for storing the sequence of word IDs of shuffled sentences. 
Sentences are seperated by `sentence_end` word ID (see [SentenceSet](data.md#dp.SentenceSet)).

## Neural Network Language Model ##

A neural network language model (NNLM) uses a neural network to model language. 
There are various approaches to building NNLMs.
The first NNLM was (\cite{yoshua2001nnlm}). 
(\cite{bengio2006neural}) concatenates $n$ context words embeddings at the input layer to form an 
input of size $n \times n_i$, where $n_i$ is the number of units per word embedding. 
This input layer takes the form of a \textit{lookup table}. We can think of it as a
weight matrix $W$ of size $N_t \times n_i$ where $N_t$ is the number of words in our vocabulary.
In the case of the billion words dataset, we have approximately 800,000 words. 
Each word is assigned a single row of weight 
matrix `W` which will serve as its embedding. These can be learned through 
backpropagation where the input of the table is a vector `x` of dimension 
`n` where each variable `x[i]` contains the index of the word at position `i` of the context.
These are used to extract all embeddings of the lookup table that correspond to the context words:
```lua
y = W[x[1]] || W[x[2]] || W[x[3]] || ... || W[x[n]]
```
where `||` concatenates its left and right vectors. The gradient can be calculated as follows:
```lua
 dy      1 for j in x  
----- =  
dW[j]    0 for j not-in x
```
which makes this layer efficient for both forward and backward propagation 
since only the `n` context words need to be queried, concatenated and updated.

The resulting concatenation of embeddings can be forwarded through 
the following parameterized hidden layers having the following form:
```lua
y = sigma(Wx + b)
```
where `sigma` is an element-wise [transfer]() function. NNLMs are often 
shallow networks having no more than 1 or 2 parameterized hidden layers 
(\cite{schwenk2005training}), (\cite{le2011structured}).

In its simplest form, the output layer uses a normalizing non-linearity like [softmax]():
```lua
                     exp(x[i])
y[i] = -------------------------------------
       exp(x[1])+...+exp(x[i])+...+exp(x[n])
```
where `n` is the size of vectors `x` and `y` (the size of the vocabulary), 
and `exp` is the exponential function.
The softmax's use of the exponential function has a tendency of increasing 
the weight of the highest input values, thus forming a kind of soft version 
of the `max` function. By dividing by the sum of the exponential of each 
variable in the vector `x`, it has a normalizing effect in that `y[1]+y[2]+..+y[n] = 1`, 
thus making it useful for generating multinomial probabilities `P(Y|X)`. 
The forward and backward propagation of this layer is extremely costly in 
terms of processing time for large vocabularies. This inefficiency is due 
to the normalization which requires calculation of all `x[i]` for `1 < i < n`. 
Faster alternatives to using a pure `softmax` exist, which will be discussed in the next section. 

The empirical risk function of the model is the ubiquitous mean negative log-likelihood (NLL):
```lua
      -log(y[1,t[1]])-log(y[2,t[2]])-...-log(y[N,t[N]])
NLL = -------------------------------------------------  
                             N
```
where `N` is the total number of examples, and `t[k]` is the target word having context `x[k]`. 
Finally, `y[k,t[k]]` is the likelihood of word `t[k]` for example `k`, 
where `y[k]` is the output of the NNLM given context `x[k]`.
To evaluate our NNLMs, we use perplexity (PPL) as this is the 
standard metric used in the field NLP for language modeling:
```lua
PPL = exp(NLL)
``` as long as the NLL and PPL use the same logarithm basis (`e` in this case).

## Output Layer ##
