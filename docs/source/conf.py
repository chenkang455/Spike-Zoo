# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme
project = 'Spike-Zoo'
copyright = '2025, Kang Chen'
author = 'Kang Chen'
release = 'v0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',   # 启用自动化文档生成
]


templates_path = ['_templates']
exclude_patterns = []


language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_css_files = ['css/custom.css']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
