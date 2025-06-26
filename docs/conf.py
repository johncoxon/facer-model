"""Configuration file for documentation."""
import os
from pyproject_parser import PyProject

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.todo",
              "sphinx.ext.coverage",
              "sphinx.ext.ifconfig",
              "sphinx.ext.viewcode",
              "sphinx.ext.napoleon",
              "numpydoc"
]

# General information about the project.
info = PyProject.load("../pyproject.toml")

# Set the basic variables
source_suffix = ".rst"
master_doc = "index"
project = "facer-model"
year = "2025"
author = "John C Coxon et al."
copyright = f"{year}, {author}"
version = release = info.project["version"].base_version

pygments_style = "trac"
templates_path = ["."]
html_use_smartypants = True
html_last_updated_fmt = "%d %b %Y"
html_split_index = True
html_sidebars = {"**": ["searchbox.html", "globaltoc.html", "sourcelink.html"]}
html_short_title = "%s v%s" % (project, version)
