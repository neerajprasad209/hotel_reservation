from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    
setup(
    name="MLOPS-PROJECT",
    version="0.0.1",
    author="Neeraj Prasad",
    packages=find_packages(),
    install_requires=requirements,
)

