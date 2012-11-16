# -*- coding: utf-8 -*-

import numpy
from matplotlib import pyplot as plt

from scikits.optimization.helpers import ForwardFiniteDifferences

class Recorder(object):
  def __init__(self, signal, function, origs):
    self.values = []
    self.signal = signal
    self.function = function
    self.origs = origs
    self.fig = plt.figure(figsize = ( 16, 10 ))
    self.orig_fig = self.fig.add_subplot(2,2,1)
    self.values_fig = self.fig.add_subplot(2,2,3)
    self.signal0_fig = self.fig.add_subplot(2,2,2)
    self.signal1_fig = self.fig.add_subplot(2,2,4)

  def __call__(self, **state):
    self.values.append(state["new_value"])
    self.parameters = state["new_parameters"]
    self.plot_estimation()
    self.plot_values()
    self.plot_signals()
    plt.draw()
    self.fig.savefig("%04i.png" % state["iteration"])

  def plot_estimation(self):
    length = len(self.parameters) / 2
    self.orig_fig.clear()
    self.orig_fig.plot(signal, label="Original")
    self.orig_fig.plot(self.function.create_estimation(self.parameters), label="Estimated")
    self.orig_fig.legend()

  def plot_values(self):
    self.values_fig.clear()
    self.values_fig.semilogy(self.values)
    self.values_fig.set_title("Cost")

  def plot_signals(self):
    length = len(self.parameters) / 2
    self.signal0_fig.clear()
    self.signal0_fig.plot(self.origs[0], label="Original")
    self.signal0_fig.plot(self.parameters[:len(self.origs[0])], label="Estimated")
    self.signal0_fig.set_title("Signal 0 (SNR %f)" % (10 * numpy.log10( numpy.sum((self.parameters[:len(self.origs[0])] - self.origs[0])**2) / numpy.sum((self.origs[0])**2))))
    self.signal0_fig.set_xlim(-1, len(self.origs[0]))
    self.signal0_fig.legend()

    self.signal1_fig.clear()
    self.signal1_fig.plot(self.origs[1], label="Original")
    self.signal1_fig.plot(self.parameters[length:length + len(self.origs[1])], label="Estimated")
    self.signal1_fig.set_title("Signal 1 (SNR %f)" % (10 * numpy.log10( numpy.sum((self.parameters[length:length + len(self.origs[1])] - self.origs[1])**2) / numpy.sum((self.origs[1])**2))))
    self.signal1_fig.set_xlim(-1, len(self.origs[1]))
    self.signal1_fig.legend()

class Function(ForwardFiniteDifferences):
  def __init__(self, signal, combs):
    ForwardFiniteDifferences.__init__(self)
    self.signal = signal
    self.combs = combs
    self.mu = 20000

  def create_estimation(self, x):
    length = len(x) / len(self.combs)
    return numpy.convolve(x[:length], self.combs[0])[:length] + numpy.convolve(x[length:], self.combs[1])[:length]

  def __call__(self, x):
    return numpy.sum((self.signal - self.create_estimation(x))**2) + numpy.sum(numpy.abs(x)) / self.mu

  def gradient(self, x):
    error = self.signal - self.create_estimation(x)
    grad = numpy.zeros(len(x))

    length = len(x) / len(self.combs)
    grad[:length] = - 2 * numpy.convolve(self.combs[0], error[::-1])[:length][::-1]
    grad[length:] = - 2 * numpy.convolve(self.combs[1], error[::-1])[:length][::-1]

    return grad + numpy.sign(x) / self.mu

def find_x0(signal, combs):
  length = len(signal)
  s0 = numpy.convolve(signal, combs[0][::-1])[:length] / numpy.sum(combs[0])
  s1 = numpy.convolve(signal, combs[1][::-1])[:length] / numpy.sum(combs[1])
  return numpy.vstack((s0, s1)).flatten()

if __name__ == "__main__":
  plt.ion()

  from scikits.optimization import *
  import pickle

  f = open("dataset")
  combs = pickle.load(f)
  signal = pickle.load(f)
  origs = pickle.load(f)

  fun = Function(signal, combs)

  mystep = step.FRPRPConjugateGradientStep()
  mylinesearch = line_search.WolfePowellRule()
  mycriterion = criterion.criterion(ftol = 0.000001, iterations_max = 500)
  myrecorder = Recorder(signal, fun, origs)
  myoptimizer = optimizer.StandardOptimizer(function = fun,
                                            step = mystep,
                                            line_search = mylinesearch,
                                            criterion = mycriterion,
                                            record = myrecorder,
#                                            x0 = find_x0(signal, combs))
                                            x0 = numpy.zeros(2*len(signal)))

  print "Starting optimization"
  xf = myoptimizer.optimize()
  length = len(signal)
  xf0 = xf[:length]
  xf1 = xf[length:]
  print xf0
  print xf1
  print myoptimizer.state['istop']

  plt.ioff()
  plt.show()
