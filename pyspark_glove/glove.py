"""
Filename: run.py
Creation Date: 6/6/18

Author: Gareth Jones

Description:
    An implementation of the GloVe algorithm in Spark. This implementation
    borrows from https://github.com/dmarcous/spark-glove and code from
    the original implementation, found here

TODOs
-----

TODO: monitor loss by returning it through the reduce step

"""

from typing import Tuple, Dict, Iterable

import numpy as np
from pyspark import RDD
from pyspark import SparkContext

Array = np.ndarray


def _initialize_parameters(unique_words: Iterable, vector_size: int) -> (
        Dict[str, Array], Dict[str, float], Dict[str, Array], Dict[str, float]):
  """Initialize the model parameters

  This requires a full pass over the data to compute the vocabulary
  of the dataset

  TODO: shorten variable names

  Parameters
  ----------
  unique_words : RDD
  vector_size

  Returns
  -------
  _vectors
  _biases
  _vector_gradients
  _bias_gradients

  """
  print('Initializing Model Parameters...')

  _vectors = {}
  _biases = {}
  _vector_gradients = {}
  _bias_gradients = {}

  # Generate parameters for each word
  for word in unique_words:
    _vectors[word] = np.random.uniform(-0.5, .5, size=vector_size) / vector_size
    _biases[word] = 0.
    _vector_gradients[word] = np.full(vector_size, 1.0)
    _bias_gradients[word] = 1.

  return _vectors, _biases, _vector_gradients, _bias_gradients


##########################
#                        #
#       Model Code       #
#                        #
##########################


def _compute_grads(loss: float, other_vector: Array, vector_grads: Array,
                   bias_grad: float) -> (
        Array, float, Array, float):
  """Function for evaluating the gradient of a word

  TODO: add latex math of grad here

  Parameters
  ----------
  loss : The loss with the learning rate applied
  other_vector : The vector of the other word, used for grad calculation
  vector_grads : The running gradient of the word vector
  bias : The bias term for the word
  bias_grad : The running gradients of the word's bias

  Returns
  -------
  new_vector_grads : computed vector gradient
  new_bias_grad : computed bias gradient
  vector_grad_update : The update to the running vector gradient
  bias_grad_update : The update to running bias gradient

  """
  weight_grad = loss * other_vector

  # Adaptive gradients
  new_vector_grads = weight_grad / np.sqrt(vector_grads)
  new_bias_grad = loss / np.sqrt(bias_grad)

  # Store gradient norm
  bias_grad_update = np.square(loss)
  vector_grad_update = np.square(weight_grad)

  return new_vector_grads, new_bias_grad, vector_grad_update, bias_grad_update


def _gradient_update(row: Tuple[Tuple[str, str], int],
                     vectors: Dict[str, Array],
                     vector_gradients: Dict[str, Array],
                     biases: Dict[str, float],
                     bias_gradients: Dict[str, float],
                     max_value: int,
                     learning_rate: float,
                     alpha: float):
  """

  Parameters
  ----------
  row : co-occurrence tuple
  vectors : Dict mapping words to their vector
  vector_gradients : Dict mapping words to the running gradient of the vector
  biases : Dict mapping words to their bias term
  bias_gradients : : Dict mapping words to the running gradient of the bias
  max_value : The maximum value where loss weighting is applied
  learning_rate : The learning rate of the vector
  alpha : Part of the loss weighting

  Returns
  -------
  Key-Value pair of word, word gradients for both words

  """
  word_0, word_1 = row[0]
  count = row[1]

  # Parameters
  word_0_vec = vectors[word_0]
  word_1_vec = vectors[word_1]

  word_0_bias = biases[word_0]
  word_1_bias = biases[word_1]

  # Gradients
  word_0_vec_grads = vector_gradients[word_0]
  word_1_vec_grads = vector_gradients[word_1]

  word_0_bias_grads = bias_gradients[word_0]
  word_1_bias_grads = bias_gradients[word_1]

  # Loss calculation
  product = word_0_vec.dot(
    word_1_vec) + word_0_bias + word_1_bias  # The regression estimate
  loss = product - np.log(count)  # loss
  weight = np.power(np.minimum(1., count / max_value), alpha)  # loss weight
  weighted_loss = weight * loss  # weighted loss calculation
  learning_loss = weighted_loss * learning_rate

  # Compute the gradients
  word_0_grads = _compute_grads(learning_loss, word_1_vec, word_0_vec_grads,
                                word_0_bias_grads)
  word_1_grads = _compute_grads(learning_loss, word_0_vec, word_1_vec_grads,
                                word_1_bias_grads)

  return [(word_0, word_0_grads), (word_1, word_1_grads)]


def train_glove(spark: SparkContext, word_cooc: RDD, num_iterations=100,
                vector_size=10, learning_rate=0.001, max_value=100,
                alpha=3. / 4) -> (Dict[str, Array], Dict[str, float]):
  """Train a glove model

  TODO: add option to initialize form existing parameters for continued training

  Parameters
  ----------
  spark : The Spark context of the session
  word_cooc :  The co-occurrence RDD of words, ([word, word], count)
  max_value :  The max value of the loss weighting. Counts higher then this do
    not have the loss applied to them
  num_iterations : the number of training iterations to run
  max_value : The maximum value where loss weighting is applied
  learning_rate : The learning rate of the vector
  alpha : Part of the loss weighting

  Returns
  -------
’
  """
  if num_iterations > 0:
    raise ValueError('The number of training iterations must be greater than 0')

  # Model Hyper-parameters
  max_value_bc = spark.broadcast(max_value)
  learning_rate_bc = spark.broadcast(learning_rate)
  alpha_bc = spark.broadcast(alpha)

  # Get the unique words to initialize the parameter dicts
  unique_words = word_cooc.keys().flatMap(lambda x: x).distinct().collect()

  # Initialize the model parameters
  init_vectors, init_biases, init_vectors_grads, init_biases_grads = _initialize_parameters(
    unique_words, vector_size)

  # Broadcast the new model params
  word_vectors = spark.broadcast(init_vectors)
  word_biases = spark.broadcast(init_biases)
  word_vector_grads = spark.broadcast(init_vectors_grads)
  word_bias_grads = spark.broadcast(init_biases_grads)

  # Start training
  for i in range(1, num_iterations + 1):
    print('Iteration Number:', i)
    print('\tComputing Gradients...')
    # Compute the loss for every word co-occurrence
    updates = word_cooc.flatMap(lambda x:
                                _gradient_update(x,
                                                 word_vectors.value,
                                                 word_vector_grads.value,
                                                 word_biases.value,
                                                 word_bias_grads.value,
                                                 max_value_bc.value,
                                                 learning_rate_bc.value,
                                                 alpha_bc.value))

    # Collect gradients and sum over words
    aggregated_grads = updates.reduceByKey(
      lambda x, y: [x[i] + y[i] for i in range(4)]).collect()
    print('\tUpdating Params')

    # Separate update components
    updated_vectors = {}
    for word, grad in [(word, grad[0]) for word, grad in aggregated_grads]:
      updated_vectors[word] = word_vectors.value[word] - grad

    updated_biases = {}
    for word, grad in [(word, grad[1]) for word, grad in aggregated_grads]:
      updated_biases[word] = word_biases.value[word] - grad

    updated_vector_grads = {}
    for word, grad in [(word, grads[2]) for word, grads in aggregated_grads]:
      updated_vector_grads[word] = word_vector_grads.value[word] + grad

    updated_bias_grads = {}
    for word, grad in [(word, grads[3]) for word, grads in aggregated_grads]:
      updated_bias_grads[word] = word_bias_grads.value[word] + grad

    # Un-persist old values
    for bc_var in [word_vectors, word_vector_grads, word_biases,
                   word_vector_grads]:
      bc_var.unpersist()

    # Broadcast updates
    word_vectors = spark.broadcast(updated_vectors)
    word_biases = spark.broadcast(updated_biases)
    word_vector_grads = spark.broadcast(updated_vector_grads)
    word_bias_grads = spark.broadcast(updated_bias_grads)

  # noinspection PyUnboundLocalVariable
  return updated_vectors, updated_biases
