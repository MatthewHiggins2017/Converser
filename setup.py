from setuptools import setup, find_packages
import glob

setup(
    name='Converser',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    scripts=glob.glob('scripts/*'),
    author='Matthew Charles William Higgins (Augural Ltd)',
    python_requires='>=3.11',
)