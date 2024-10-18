"""Script to make repository installable."""

from distutils.core import setup

from setuptools import find_packages

setup(name="MalamuteCode",
      version="1.0",
      description="MalamuteCode repository.",
      author="MalamuteCode",
      packages=find_packages("src"),
      package_dir={
          "": "src",
          "test": "test"
      },
      py_modules=["logger"],
      )
