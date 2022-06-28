from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='juliet_mods',
    version='0.5',
    description='A collection of convenience functions for juliet',
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jonas Kemmer @ ZAH, Landessternwarte Heidelberg',
    url="https://github.com/JonasKemmer/JulietMods",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'matplotlib', 'corner', 'pandas', 'seaborn', 'astropy'
    ],
    python_requires='>=3.6',
)
