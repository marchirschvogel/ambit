# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ambit'
copyright = '2023, Marc Hirschvogel'
author = 'Marc Hirschvogel'
release = '1.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", # to have Markdown working with Sphinx...
              "sphinxcontrib.bibtex",
              "sphinx.ext.autodoc",
              "sphinx.ext.autosummary"]

autodoc_mock_imports = ["mpi4py","petsc4py","dolfinx","basix"]

bibtex_bibfiles = ['ref.bib']

templates_path = ['_templates']
exclude_patterns = []

latex_elements = {
    'preamble': r'''
\mathchardef\mhyphen="2D
''',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
