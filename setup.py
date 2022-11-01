from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

setup(
    name='kissdsp',
    version='1.0.0',
    description='Keep It Simple Stupid - Digital Signal Processing',
    author='Francois Grondin',
    author_email='francois.grondin2@usherbrooke.ca',
    url='https://github.com/francoisgrondin/kissdsp',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'matplotlib',
        'numpy',
        'rir_generator',
        'soundfile'
    ]
)