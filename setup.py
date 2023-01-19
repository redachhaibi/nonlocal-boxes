from setuptools import Extension, setup, find_packages
from os import path

local_path = path.abspath(path.dirname(__file__))

print("Local path: ", local_path)
print("")

print("Launching setup...")
# Setup
setup(
    name='non_local_boxes',

    version='0.01',

    description='',
    long_description=""" 
    """,
    url='',

    author='Anonymous',
    author_email='Anonymous',

    license='MIT License',

    install_requires=["numpy", "matplotlib", "scipy", "torch"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },

    ext_modules=[],
)