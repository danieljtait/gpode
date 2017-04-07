import imp

home = '/Users/danieltait/'
foo = imp.load_source('gp',home + '/MyGits/GPODE/GPODE/gp/__init__.py')
fo1 = imp.load_source('gp_ode',home + '/MyGits/GPODE/GPODE/gp_ode/__init__.py')

from gp import *
from gp_ode import *


from .core import GaussianProcess, ksqexp
