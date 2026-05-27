import os
import sys

project = "dyphur"
author = "Alexander Solovets"
copyright = "2025, Alexander Solovets"
release = "0.0.1"
version = "0.0.1"

extensions = [
    "breathe",
    "sphinx_rtd_theme",
]

breathe_projects = {"dyphur": os.path.join(os.path.dirname(__file__), "doxygen-xml/xml")}
breathe_default_project = "dyphur"
breathe_default_members = ("members", "undoc-members")

html_theme = "sphinx_rtd_theme"
html_static_path = []

exclude_patterns = ["_build", "doxygen-xml", "Thumbs.db", ".DS_Store"]
