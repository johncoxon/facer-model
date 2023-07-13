import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(author="John Coxon and Steve Milan",
                 author_email="work@johncoxon.co.uk",
                 classifiers=[
                    "Development Status :: 4 - Beta",
                    "Intended Audience :: Science/Research",
                    "Natural Language :: English",
                    "Programming Language :: Python :: 3",
                 ],
                 description="A Python implementation of the Birkeland current model "
                             "presented by Milan (2013).",
                 install_requires=[
                     "numpy",
                     "pandas"
                 ],
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 name="birkeland",
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 python_requires=">=3.9",
                 url="https://github.com/johncoxon/birkeland",
                 version="1.0",
                 )
