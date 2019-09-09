import setuptools

long_description = '''
This package extends the python-mwlinks toolkit released by mediawiki. 

Extracts links from MediaWiki with a focus on Wikipedia.

This library add multi-process utilities for extracting wikilinks from MediaWiki XML database dumps.
'''

setuptools.setup(
    name="python-mwlinks",
    version="0.0.1",
    url="https://github.com/hunterhector/python-mwlinks",

    description="A python MediaWiki Link parser",
    long_description=long_description,

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'mwxml',
        'mwtypes',
        'docopt',
        'jsonable',
    ],
    extras_require={
    },
    package_data={
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
