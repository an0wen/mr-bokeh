import os
from .mrbokeh import *
from .moustache import *

__version__ = "1.O.6"

# set Python env variable to keep track of example data dir
mrbokeh_dir = os.path.dirname(__file__)
DATADIR = os.path.join(mrbokeh_dir, "example_data/")