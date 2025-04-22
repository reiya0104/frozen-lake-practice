lint:
	poetry run mypy . --explicit-package-bases
	poetry run ruff check . --fix

format:
	poetry run ruff format . --in-place


