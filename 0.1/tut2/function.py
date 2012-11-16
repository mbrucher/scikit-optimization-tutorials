
import numpy

class Function(object):
    def __call__(self, x):
        t = numpy.sqrt(numpy.sum(x**2, axis=0))
        return -numpy.sinc(t)
    def gradient(self, x):
        t = numpy.sqrt(numpy.sum(x**2, axis=0))
        return x / t**2 * (-numpy.cos(numpy.pi*t) + numpy.sinc(t))
    
def get_space_function():
  fun = Function()
  d = numpy.linspace(-3, 3, num=99)
  X, Y = numpy.meshgrid(d, d)
  Z = fun(numpy.array((X,Y)))
  return X, Y, Z
