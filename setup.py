from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='MLOPS-PRJECT-5',
    version='0.1',
    author='keerthi kumar',
    install_requires=requirements,
    packages=find_packages()
)