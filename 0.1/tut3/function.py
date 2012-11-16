
import numpy

from scikits.optimization.helpers import ForwardFiniteDifferences

class Function(ForwardFiniteDifferences):
  def __init__(self):
    ForwardFiniteDifferences.__init__(self)
    self.a = numpy.array((1, 2))
  def __call__(self, x):
    t = numpy.sqrt(numpy.sum((self.a * x)**2))
    return -numpy.sinc(numpy.pi * t)
   
def get_space_function():
  fun = Function()
  d = numpy.linspace(-.5,.5, num=99)
  X, Y = numpy.meshgrid(d, d)
  Z = numpy.zeros((99,99))
  for i in range(99):
    for j in range(99):
      Z[i,j] = fun((X[i,j], Y[i,j]))
  return X, Y, Z
