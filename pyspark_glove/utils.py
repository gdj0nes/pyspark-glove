"""
Filename: run.py
Creation Date: 6/6/18

Author: Gareth Jones

Description:
    An implementation of the GloVe algorithm in Spark. This implementation borrows from
    https://github.com/dmarcous/spark-glove and code from the original implementation, found here

TODOs
-----

"""
from itertools import combinations

import numpy as np
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, StringType, FloatType


def _co_occurences(words):
  """Produce co-occurrences from

  TODO: might need to add for case where there is no other item, None item ?
  TODO: implement windowing for text data

  Parameters
  ----------
  words : Column

  Returns
  -------
  coocs :

  """
  if len(words) == 1:
    return []
  else:
    return list(combinations(words, 2))


co_occurences_udf = fn.udf(_co_occurences, ArrayType(ArrayType(StringType())))


def _sentence_embedding(words, vectors):
  embeddings = []
  for word in words:
    try:
      embeddings.append(vectors[word])
    except KeyError:
      # Generate a random embedding
      vec_len = len(vectors[list(vectors.keys())[0]])
      embeddings.append(np.random.uniform(-0.5, .5, size=vec_len) / vec_len)

  embedding_vec = np.vstack(embeddings).sum(axis=0)
  embedding_vec /= np.sqrt(
    len(words))  # Reduce likelihood that number of words is defining

  return [float(x) for x in list(embedding_vec)]

sentence_embedding_udf = fn.udf(_sentence_embedding, ArrayType(FloatType()))

