venv:
	python3 -m venv venv

activate:
	. venv/bin/activate

install:
	pip install -r requirements.txt

precommit:
	pre-commit install

setup: venv activate install precommit

clean:
	rm -rf venv

.PHONY: venv activate install precommit setup clean
