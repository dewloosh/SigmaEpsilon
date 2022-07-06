# -*- coding: utf-8 -*-
import setuptools
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


with open("README.md", "r") as fh:
    long_description = fh.read()


with open('requirements.txt') as f:
    required = f.read().splitlines()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="sigmaepsilon",                     
    version=get_version("src/sigmaepsilon/__init__.py"),                        
    author="dewloosh",
    author_email = 'dewloosh@gmail.com',
    url = 'https://github.com/dewloosh/sigmaepsilon',   
    download_url = 'https://github.com/dewloosh/sigmaepsilon/releases',                     
    keywords = ['Finite Element Method', 'FEM', 'FEA', 'CAD', 'Civil Engineering', 'Structural Engineering'],
    description="Computational Mechanics in Python.",
    long_description=long_description,   
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where='src'),   
    classifiers=[
        'Development Status :: 3 - Alpha',     
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',
		'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',                             
    package_dir={'':'src'},
	install_requires=required
)

