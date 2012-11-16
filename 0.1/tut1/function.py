#!/usr/bin/env python

import numpy

class Function(object):
  def __call__(self, x):
    t = numpy.sqrt((x[0]*x[0]+x[1]*x[1]))
    return -numpy.sinc(t)

def get_space_function():
  fun = Function()
  d = numpy.linspace(-3, 3, num=99)
  X, Y = numpy.meshgrid(d, d)
  Z = fun((X,Y))
  return X, Y, Z

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D

  X, Y, Z = get_space_function()
  
  plt.figure()
  plt.imshow(Z, interpolation='bilinear', origin='lower',
                cmap=cm.jet, extent=(-3,3,-3,3))
  
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)

  plt.show()
  