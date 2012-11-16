#!/usr/bin/env python

from display_with_path import add_path, create_figure
import matplotlib.pyplot as plt

class Recorder(object):
  def __call__(self, **state):
    fig = create_figure()
    add_path(fig, state['new_parameters'])
    plt.savefig('%04i.png' % state['iteration'])

if __name__ == "__main__":
  from scikits.optimization.optimizer import PolytopeOptimizer
  from scikits.optimization import criterion
  
  from function import Function
  
  import numpy

  optimizer = PolytopeOptimizer(function = Function(),
                                x0 = numpy.random.randn(3, 2),
                                criterion = criterion.OrComposition(criterion.RelativeParametersCriterion(0.00001), criterion.IterationCriterion(20)),
                                record = Recorder())
  optimizer.optimize()
  print numpy.mean(optimizer.state['new_parameters'], axis=0)
  