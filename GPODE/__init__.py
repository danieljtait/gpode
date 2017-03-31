print "I have been initialised!"

import imp

home = '/Users/danieltait/'
foo = imp.load_source('gp',home + '/MyGits/GPODE/GPODE/gp/__init__.py')

from gp import *


