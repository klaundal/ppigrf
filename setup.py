import setuptools

# Howto
#
#This is how to upload a project to pipy. This is mainly a note to self, as this is the first project I have uplo#aded to pypi:
#
#python3 -m venv /tmp/venv
#source /tmp/venv/bin/activate
#pip install build
#pip install twine
#python3 -m build
#python3 -m twine upload dist/ppigrf-1.0.0.tar.gz 

#Genrate long description using readme file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ppigrf",
    version="1.0.2",
    author="Karl Laundal",
    author_email="readme@file.md",
    description="Pure Python IGRF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        'numpy>=0.13.1',
        'pandas>=1.3.5'
    ],
    package_dir={"": "src"},
    package_data={'':['IGRF13.shc']},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)