try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'longmantide',
    'author': 'John R. Leeman',
    'url': 'Project URL https://github.com/jrleeman/LongmanTide',
    'download_url': 'https://github.com/jrleeman/LongmanTide',
    'author_email': 'kd5wxb@gmail.com',
    'version': '0.1',
    'install_requires': ['numpy', 'matplotlib'],
    'packages': ['longmantide'],
    'scripts': [],
    'name': 'LongmanTide'
}

setup(**config)
