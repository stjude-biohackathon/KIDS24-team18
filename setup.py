from setuptools import setup, find_packages
import os

# Function to read version from version.py
def get_version():
    version = {}
    with open(os.path.join('grumpy', 'version.py')) as f:
        exec(f.read(), version)
    return version['__version__']


# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='grumpy',  # Replace with your package's name
    version=get_version(),
    author='Wojciech Rosikiewicz',  # Replace with your name or organization
    author_email='Wojciech.Rosikiewicz@STJUDE.ORG',  # Replace with your email
    description='A Python package for analysis and evaluation using grumpy model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stjude-biohackathon/KIDS24-team18',  # Replace with your project's URL
    packages=find_packages(where='grumpy'),  # Finds the 'grumpy' package
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust license if needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[  # Dependencies from requirements.txt
        line.strip() for line in open('requirements.txt')
    ],
    package_data={
        # Include any data files from the 'grumpy' package
        'grumpy': ['data/*', 'templates/*'],
    },
    entry_points={
        'console_scripts': [
            'grumpy=grumpy.grumpy:main',  # Entry point for your script
        ],
    },
    include_package_data=True,
    license="MIT",  # Adjust license type if needed
)
