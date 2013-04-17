#!/usr/bin/env python

import numpy as np

from scikits.optimization import *

def create_background():
  fun = Rosenbrock()
  x = np.linspace(-1.5, 1.5, 301)

  data = np.zeros((301,301), dtype=np.float)

  for i in range(301):
    for j in range(301):
      data[i, j] = fun(np.array((x[j], x[i])))

  return data

class Rosenbrock(object):
  """
  The Rosenbrock function
  """
  def __init__(self):
    self.count = 0
    self.gradient_count = 0
  def __call__(self, x):
    """
    Get the value of the Rosenbrock function at a specific point
    """
    self.count = self.count+1
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1. - x[:-1])**2.0)

  def gradient(self, x):
    """
    Evaluates the gradient of the function
    """
    self.gradient_count = self.gradient_count+1
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros(x.shape, x.dtype)
    der[1:-1] = 200. * (xm - xm_m1**2.) - 400. * (xm_p1 - xm**2.) * xm - 2. * (1. - xm)
    der[0] = -400. * x[0] * (x[1] - x[0]**2.) - 2. * (1. - x[0])
    der[-1] = 200. * (x[-1] - x[-2]**2.)
    return der

fun = Rosenbrock()

class Recorder(object):
  def __init__(self):
    self.old = []
    self.new = []
    self.new_param = []
    self.counts = []
    self.gradient_counts = []
    
  def __call__(self, **state):
    if "old_value" in state:
      self.old.append(state["old_value"])
    self.new.append(state["new_value"])
    self.new_param.append(state["new_parameters"])
    self.counts.append(fun.count)
    self.gradient_counts.append(fun.gradient_count)

recorder = Recorder()
    
mycriterion = criterion.criterion(ftol = 0.0001, iterations_max = 10000)

startPoint = np.empty((3, 2), np.float)
startPoint[:,0] = -1.
startPoint[:,1] = 1.
startPoint[1,0] += .1
startPoint[2,1] -= .1

myoptimizer = optimizer.PolytopeOptimizer(function = fun,
                                          criterion = mycriterion,
                                          x0 = startPoint,
                                          record = recorder
                                          )

print myoptimizer.optimize()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(-1, -1 + len(recorder.new)), recorder.counts, label="Calls")
plt.plot(np.arange(-1, -1 + len(recorder.new)), recorder.gradient_counts, label="Gradient")
plt.legend()
plt.title("Rosenbrock calls for the Simplex")
plt.savefig("ros_simplex_calls.png")

plt.figure()
plt.semilogy(recorder.old, label="Old value")
plt.semilogy(np.arange(-1, -1 + len(recorder.new)), np.array(recorder.new) + 1e-40, label="New value")
plt.legend()
plt.title("Rosenbrock values with the Simplex")
plt.savefig("ros_simplex_values.png")

plt.figure()
plt.imshow(create_background(), extent=(-1.5,1.5,1.5,-1.5))
plt.colorbar()
list_params = np.array(recorder.new_param)
plt.scatter(list_params[:,0], list_params[:,1])
plt.title("Rosenbrock positions with the Simplex")
plt.savefig("ros_simplex.png")

plt.show()
