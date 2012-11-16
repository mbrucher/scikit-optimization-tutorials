# -*- coding: utf-8 -*-

import numpy

class OrthogonalSequences(object):
  def __init__(self, size, number):
    self.size = size
    self.number = number
    self.genome_length = size * number

  @staticmethod
  def convert_slice_to_time(chromosome):
    """
    Converts a chromosome in a time sequence
    """
    total = numpy.cumsum(chromosome)
    array = numpy.zeros(total[-1]+1)
    array[total] = 1
    array[0] = 1
    return array

  def evaluate(self, chromosomes):
    """
    Evaluate the cost (square of error) of the genome by evaluating cross- and autocorrelation.
    """
    value = 0
    for i in range(self.number):
      slice_i = self.convert_slice_to_time(chromosomes[i * self.size:(i+1) * self.size])
      value += .1 * len(slice_i)
      for j in range(i, self.number):
        slice_j = self.convert_slice_to_time(chromosomes[j * self.size:(j+1) * self.size])
        correlation = numpy.correlate(slice_i, slice_j, mode = "full")
        if (i == j):
          correlation[len(slice_i)-1] -= self.size

        value += numpy.sum(correlation ** 2)

    return value

  def plot(self, chromosomes):
    """
    Plots the different sequences and the cross- and autocorrelation entries
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()

    for i in range(self.number):
      slice_i = self.convert_slice_to_time(chromosomes[i * self.size:(i+1) * self.size])
      for j in range(i, self.number):
        slice_j = self.convert_slice_to_time(chromosomes[j * self.size:(j+1) * self.size])
        ax = fig.add_subplot(self.number, self.number, (i+1) + j * self.number)
        ax.plot(numpy.arange(-len(slice_i)+1, len(slice_j)), numpy.correlate(slice_i, slice_j, mode = "full"))
    fig.suptitle("Correlation between combs")

    for i in range(self.number):
      fig = plt.figure()
      fig.gca().stem(chromosomes[i * self.size:(i+1) * self.size], numpy.ones(self.size))
      fig.gca().stem((0,), (1,))
    plt.show()

from numpy.testing import assert_array_equal

class TestChromosome(object):
  def test_convert(self):
    chromosome = (2, 3, 4, 5)
    array = OrthogonalSequences.convert_slice_to_time(chromosome)
    assert_array_equal(array, (1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1))

if __name__ == "__main__":
  from pyevolve import G1DList, GSimpleGA, Consts
  from pyevolve.DBAdapters import DBSQLite
  import sys

  size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
  size -= 1
  number = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  rangemax = int(sys.argv[3]) if len(sys.argv) > 3 else 100
  generations = int(sys.argv[4]) if len(sys.argv) > 4 else 50
  population = int(sys.argv[5]) if len(sys.argv) > 5 else 100

  genome = G1DList.G1DList(number*size)
  sequences = OrthogonalSequences(size, number)
  ga = GSimpleGA.GSimpleGA(genome)

  genome.setParams(rangemin=1, rangemax=rangemax)
  genome.evaluator.set(sequences.evaluate)

  sqlite_adapter = DBSQLite(identify="ex1", resetDB=True)
  ga.setDBAdapter(sqlite_adapter)
  ga.setMinimax(Consts.minimaxType["minimize"])

  ga.setPopulationSize(population)
  ga.setElitism(True)
  ga.setGenerations(generations)
  ga.evolve(freq_stats = generations/10)
  print ga.bestIndividual()

  for i in range(number):
    print "Comb %i: " % i, sequences.convert_slice_to_time(ga.bestIndividual()[i * size:(i+1) * size])

  sequences.plot(ga.bestIndividual())
