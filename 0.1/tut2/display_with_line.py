#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy

def add_line(fig, path):
  """
  Adds a line on the figure
  """
  import matplotlib.path as mpath
  import matplotlib.patches as mpatches

  pathdata = [(mpath.Path.MOVETO, path[0])]
  for el in path[1:]:
    pathdata.append((mpath.Path.LINETO, el))

  codes, verts = zip(*pathdata)
  path = mpath.Path(verts, codes)
  patch = mpatches.PathPatch(path, facecolor='none', edgecolor='green', alpha=0.5)
  fig.gca().add_patch(patch)

def create_figure():
  import matplotlib.cm as cm
  from function import get_space_function

  X, Y, Z = get_space_function()
  fig = plt.figure()
  plt.imshow(Z, interpolation='bilinear', origin='lower',
                cmap=cm.jet, extent=(-3,3,-3,3))
  return fig
  
if __name__ == "__main__":
  import numpy

  fig = create_figure()
  path = numpy.random.randn(3, 2) * .5
  add_line(fig, path)
  
  plt.show()
