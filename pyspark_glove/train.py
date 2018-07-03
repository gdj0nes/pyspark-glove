"""
Filename: train.py
Creation Date: 6/6/18

Author: Gareth Jones

Description:

TODOs
-----
TODO: explore the effectiveness of regularization

"""

import argparse

import numpy as np
from pyspark import SQLContext, SparkConf, SparkContext

from pyspark_glove.glove import train_glove
from pyspark_glove.utils import co_occurences_udf

Array = np.ndarray


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num_iterations', default=20, type=int)
  parser.add_argument('-s', '--vector_size', default=10, type=int)
  parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
  parser.add_argument('-i', '--input_path', type=str)
  parser.add_argument('-c', '--word_column', type=str, default='words')
  parser.add_argument('-m', '--max_value', type=int, default=100)
  parser.add_argument('-a', '--alpha', type=float, default=3. / 4.)

  args = parser.parse_args()

  # Spark initialization
  conf = SparkConf().setAll(
    [('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')])
  sc = SparkContext(appName="GloVe", conf=conf)
  sqlctx = SQLContext(sc)
  sc.setLogLevel('warn')

  # Make co-occurence dataset
  raw_data = sqlctx.read.parquet(args.input_path)
  cooc_entries = (
    raw_data
      .select(co_occurences_udf('words').alias('co_occurences'))
      .rdd
      .flatMap(lambda x: (x, 1))
      .keyBy(lambda r: "|".join(list(sorted(r[0]))))  # Key by words
      .reduceByKey(lambda x, y: x + y)
  )

  word_vectors, word_biases = train_glove(spark=sc,
                                          word_cooc=cooc_entries,
                                          max_value=args.max_value,
                                          num_iterations=args.num_iterations,
                                          learning_rate=args.learning_rate,
                                          vector_size=args.vector_size,
                                          alpha=args.alpha)
  sc.stop()

  return word_vectors, word_biases


if __name__ == '__main__':
  main()
