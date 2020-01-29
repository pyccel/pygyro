from setuptools import setup, find_packages

import pygyro

#==============================================================================

NAME     = 'pygyro'
VERSION  =  pygyro.__version__
AUTHOR   = 'Emily Bourne, Yaman Güçlü'
EMAIL    = 'yaman.guclu@gmail.com'
URL      = 'https://github.com/pyccel/pygyro'
DESCR    = 'Python library for parallel gyro-kinetic simulations'
KEYWORDS = ['Semi-Lagrangian', 'MPI', 'Gyrokinetic']
#LICENSE  = 'LICENSE.txt'

#==============================================================================

def setup_package():
    setup(
        name             = NAME,
        version          = VERSION,
        author           = AUTHOR,
        author_email     = EMAIL,
        description      = DESCR,
        long_description = open('README.md').read(),
    #    license          = LICENSE,
        keywords         = KEYWORDS,
        url              = URL,
        packages         = find_packages(),
        classifiers      = [
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
    #        "License :: OSI Approved :: BSD License",
        ],
    )

#..............................................................................
if __name__ == '__main__':
    setup_package()
