# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = open("README.md").read()

setup(
    name="septum_mec",
    packages=find_packages(),
    version='0.1',
    include_package_data=True,
    author="Mikkel Elle Lepperød",
    author_email="m.e.lepperod@medisin.uio.no",
    maintainer="Mikkel Elle Lepperød",
    maintainer_email="m.e.lepperod@medisin.uio.no",
    platforms=['Linux', "Windows"],
    description="Plugins for the septum-mec project",
    long_description=long_description,
)
