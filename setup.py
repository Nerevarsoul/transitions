import codecs
import sys
from setuptools import setup, find_packages
from setuptools.command.install_lib import install_lib
from setuptools.command.build_py import build_py

with open('transitions/version.py') as f:
    exec(f.read())

with codecs.open('README.md', 'r', 'utf-8') as f:
    # cut the badges from the description and also the TOC which is currently not working on PyPi
    long_description = ''.join(f.readlines()[49:])

if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

tests_require = ['dill', 'pygraphviz']
extras_require = {'diagrams': ['pygraphviz']}

extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    extras_require['test'] = ['nose>=0.10.1']
    tests_require.append('nose')
    extra_setuptools_args = dict(
        test_suite='nose.collector',
    )


def _not_async(filepath):
    return filepath.find('aio/') < 0


# Do not copy async module for Python 3.4 or below.
class nocopy_async(build_py):
    def find_all_modules(self):
        modules = build_py.find_all_modules(self)
        modules = list(filter(lambda m: _not_async(m[-1]), modules))
        return modules

    def find_package_modules(self, package, package_dir):
        modules = build_py.find_package_modules(self, package, package_dir)
        modules = list(filter(lambda m: _not_async(m[-1]), modules))
        return modules


# Do not compile async.py for Python 3.4 or below.
class nocompile_async(install_lib):
    def byte_compile(self, files):
        files = list(filter(_not_async, files))
        install_lib.byte_compile(self, files)


PY_35 = sys.version_info >= (3,5)
cmdclass = {}

if not PY_35:
    # do not copy/compile async version for older Python
    cmdclass['build_py'] = nocopy_async
    cmdclass['install_lib'] = nocompile_async

setup(
    cmdclass=cmdclass,
    name="transitions",
    version=__version__,
    description="A lightweight, object-oriented Python state machine implementation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tal Yarkoni',
    author_email='tyarkoni@gmail.com',
    maintainer='Alexander Neumann',
    maintainer_email='aleneum@gmail.com',
    url='http://github.com/pytransitions/transitions',
    packages=find_packages(exclude=['tests', 'test_*']),
    package_data={'transitions': ['data/*'],
                  'transitions.tests': ['data/*']
                  },
    include_package_data=True,
    install_requires=['six'],
    extras_require=extras_require,
    tests_require=tests_require,
    license='MIT',
    download_url='https://github.com/pytransitions/transitions/archive/%s.tar.gz' % __version__,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    **extra_setuptools_args
)
