# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/dewloosh'))

# -- Project information -----------------------------------------------------

project = 'dewloosh.math'
copyright = '2022, Bence Balogh'
author = 'Bence Balogh'

# The full version, including alpha/beta/rc tags
#release = '2022'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # allows to work with markdown files
    'myst_parser',  # pip install myst-parser for this

    # to plot summary about durations of file generations
    # 'sphinx.ext.duration',

    # to test code snippets in docstrings
    # 'sphinx.ext.doctest',

    # for automatic exploration of the source files
    'sphinx.ext.autodoc',

    # to enable cross referencing other documents on the internet
    # 'sphinx.ext.intersphinx',

    # Napoleon is a extension that enables Sphinx to parse both NumPy and Google style docstrings
    'sphinx.ext.napoleon',

    'nbsphinx',  # to handle jupyter notebooks
    'nbsphinx_link',  # for including notebook files from outside the sphinx source root

    'sphinx_copybutton',  # for "copy to clipboard" buttons
    'sphinx.ext.mathjax',  # for math equations
    'sphinxcontrib.bibtex',  # for bibliographic references
    'sphinxcontrib.rsvgconverter',  # for SVG->PDF conversion in LaTeX output
    'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)

    # 'sphinx.ext.coverage',
]

# set up InterSphinx mapping
#intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']
#source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The master toctree document.
master_doc = 'index'


# --------- nbsphinx-related settings ------------

# Default language for syntax highlighting in reST and Markdown cells:
highlight_language = 'none'

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ''

# Work-around until https://github.com/sphinx-doc/sphinx/issues/4229 is solved:
html_scaled_image_link = False

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Environment variables to be passed to the kernel:
#os.environ['MY_DUMMY_VARIABLE'] = 'Hello from conf.py!'

# nbsphinx_thumbnails = {
#    'gallery/thumbnail-from-conf-py': 'gallery/a-local-file.png',
#    'gallery/*-rst': '_static/copy-button.svg',
# }

# Ensure env.metadata[env.docname]['nbsphinx-link-target']
# points relative to repo root:
nbsphinx_link_target_root = repo

# This is processed by Jinja2 and inserted before each notebook

nbsphinx_prolog = (
r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'] %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='docs/source/') %}
{% endif %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        This page was generated from `{{ docpath }}`__.
    __ https://github.com/vidartf/nbsphinx-link/blob/
        """ +
git_rev + r"{{ docpath }}"
)

nbsphinx_prolog = r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'] %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='docs/source/') %}
{% endif %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{ docname }}`__.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/dewloosh/dewloosh-math/{{ env.config.release }}?filepath={{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/dewloosh/dewloosh-math/blob/{{ env.config.release }}/{{ docname }}

"""

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']]
    },
    'svg': {
        'fontCache': 'global'
    }
}

# Additional files needed for generating LaTeX/PDF output:
#latex_additional_files = ['references.bib']

# Support for notebook formats other than .ipynb
#nbsphinx_custom_formats = {
#    '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
#}

# -- The settings below this line are not specific to nbsphinx ------------

linkcheck_ignore = [r'http://localhost:\d+/']

# -- Get version information and date from Git ----------------------------

try:
    from subprocess import check_output
    release = check_output(['git', 'describe', '--tags', '--always'])
    release = release.decode().strip()
    today = check_output(['git', 'show', '-s', '--format=%ad', '--date=short'])
    today = today.decode().strip()
except Exception:
    release = '<unknown>'
    today = '<unknown date>'
    
# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'nbsphinx-linkdoc'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
#html_title = "dewloosh.math"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

html_title = project + ' version ' + release

# -- Options for LaTeX output ---------------------------------------------

# See https://www.sphinx-doc.org/en/master/latex.html
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'dewloosh-math.tex', project, author, 'Manual'),
]

latex_show_urls = 'footnote'
latex_show_pagerefs = True

# -- Options for EPUB output ----------------------------------------------

# These are just defined to avoid Sphinx warnings related to EPUB:
version = release
suppress_warnings = ['epub.unknown_project_files']
