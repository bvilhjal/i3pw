"""Put the repository root on ``sys.path`` for the test session.

``i3pw`` itself is imported from the installed (editable) distribution, but
``benchmarks/`` is deliberately not part of the package -- it is evidence-generating
scaffolding, not API -- and ``tests/test_benchmarks.py`` still has to import it.
pytest prepends the directory holding this file, so the bare ``import benchmarks``
in that module resolves without a src-layout install or a path hack in the test.
"""
