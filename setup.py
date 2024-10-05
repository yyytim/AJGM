from setuptools import setup, find_packages

setup(
    name='AJGM',
    version='1.0',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn'],
    author='stevenyang',
    author_email='yangsq@hnu.edu.cn'
)