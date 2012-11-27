#!/usr/bin/env python

import numpy as np

from scikits.optimization import criterion, step, optimizer, line_search
import matplotlib.pyplot as plt

class Rosenbrock:
  """
  The Rosenbrock function
  """
  def __call__(self, x):
    """
    Get the value of the Rosenbrock function at a specific point
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1. - x[:-1])**2.0)

def create_background():
  fun = Rosenbrock()
  x = np.linspace(0, 1.5, 151)

  data = np.zeros((151,151), dtype=np.float)

  for i in range(151):
    for j in range(151):
      data[i, j] = fun(np.array((x[j], x[i])))

  return data

class Recorder(object):
  def __init__(self):
    self.data = create_background()

  def __call__(self, **state):
    plt.close('all')
    plt.imshow(self.data, extent=(0, 1.5, 1.5, 0))
    plt.hold(True)
    plt.scatter(state["polytope_parameters"][:,0], state["polytope_parameters"][:,1], c='r')
    plt.title("Simplex state, iteration %i" % state["iteration"])
    plt.savefig("movie_%02i.png" % state["iteration"])

def start_optimization():
  startPoint = np.empty((3, 2), np.float)
  startPoint[:,0] = 1.
  startPoint[:,1] = 0.
  startPoint[1,0] -= .1
  startPoint[2,1] += .1
  recorder = Recorder()
  optimi = optimizer.PolytopeOptimizer(function = Rosenbrock(), criterion = criterion.OrComposition(criterion.AbsoluteValueCriterion(.0001), criterion.IterationCriterion(100)), x0 = startPoint, record=recorder)

  optimi.optimize()

if __name__ == "__main__":
  start_optimization()
