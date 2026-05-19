install:
	pip install -e ".[dev]"

test:
	pytest

inspect:
	ind-vias-inspect
