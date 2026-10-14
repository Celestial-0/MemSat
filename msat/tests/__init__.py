"""Test package marker.

Present so `tests.conftest` is imported exactly once, under its full name, rather
than twice — once as the top-level `conftest` pytest loads and once as
`tests.conftest` when a test module imports a helper from it. Two module objects
means two copies of every fixture, which is the kind of subtlety that produces
"the fixture works in one file and not another".

Requires `pythonpath = ["."]` in `[tool.pytest.ini_options]` so the project root
is importable.
"""
