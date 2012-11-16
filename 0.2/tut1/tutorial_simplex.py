#!/usr/bin/env python

import numpy as np

from traits.api import HasTraits, Int, Range, List, Instance, on_trait_change
from traitsui.api import View, Item, RangeEditor, Group

from chaco.api import ArrayPlotData, Plot, jet, ColorBar, HPlotContainer, LinearMapper, DataRange1D
from enable.api import Component, ComponentEditor

from scikits.optimization import criterion, step, optimizer, line_search

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

class SimplexView(HasTraits):
  """
  Display an interactive plot to visualize a simplex during an optimization
  """

  high = Int
  iterations = Range(-1, 1000)
  plot = Instance(Component)
  plotdata = Instance(ArrayPlotData)

  data = List

  view = View(
          Group(
            Item('plot', editor = ComponentEditor(), show_label=False),
            Item('iterations', editor = RangeEditor(mode="slider", is_float=False, low=-1, high_name='high'), show_label=False),
                ),
              resizable=True,
            )

  def _plotdata_default(self):
    plotdata = ArrayPlotData(values_x=self.data[0][:,0], values_y=self.data[0][:,1], background=create_background())

    return plotdata

  def _plot_default(self):
    plot = Plot(self.plotdata)
    plot.title = "Simplex on the Rosenbrock function"

    plot.img_plot("background",
                  name="background",
                  xbounds=(0,1.5),
                  ybounds=(0,1.5),
                  colormap=jet(DataRange1D(low=0,high=100)),
                  )

    plot.plot(("values_x", "values_y"), type="scatter", color="red")

    background = plot.plots["background"][0]

    colormap = background.color_mapper
    colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                        color_mapper=colormap,
                        plot=background,
                        orientation='v',
                        resizable='v',
                        width=30,
                        padding=20)
    colorbar.padding_top = plot.padding_top
    colorbar.padding_bottom = plot.padding_bottom

    container = HPlotContainer(use_backbuffer = True)
    container.add(plot)
    container.add(colorbar)
    container.bgcolor = "lightgray"
    return container

  def _high_default(self):
    return len(self.data) - 2

  @on_trait_change("iterations")
  def update_scatter(self):
    self.plotdata.set_data("values_x", self.data[self.iterations+1][:,0])
    self.plotdata.set_data("values_y", self.data[self.iterations+1][:,1])

class Recorder(HasTraits):
  view = Instance(SimplexView)

  def __call__(self, **state):
    self.view.data.append(state["polytope_parameters"])

def start_optimization(view):
  startPoint = np.empty((3, 2), np.float)
  startPoint[:,0] = 1.
  startPoint[:,1] = 0.
  startPoint[1,0] -= .1
  startPoint[2,1] += .1
  recorder = Recorder()
  recorder.view = view
  optimi = optimizer.PolytopeOptimizer(function = Rosenbrock(), criterion = criterion.OrComposition(criterion.AbsoluteValueCriterion(.0001), criterion.IterationCriterion(100)), x0 = startPoint, record=recorder)

  optimi.optimize()


if __name__ == "__main__":
  simplex = SimplexView()
  start_optimization(simplex)
  simplex.configure_traits()
