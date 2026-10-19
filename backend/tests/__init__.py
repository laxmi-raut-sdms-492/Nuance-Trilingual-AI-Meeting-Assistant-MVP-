"""
Marks backend/tests as a real package.

Without this file the directory is only a namespace portion, and Python
resolves a regular package of the same name anywhere on sys.path ahead of it.
pyannote.pipeline — pulled in by pyannote.audio — installs a top-level package
called `tests` into site-packages, so every `from tests.<module> import ...`
in this suite resolved there instead of here and collection failed outright
with ModuleNotFoundError on modules sitting in this very directory.

An __init__.py makes this a regular package, and conftest.py prepends the
backend directory to sys.path, so this one wins.
"""
