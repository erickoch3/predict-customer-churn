initvenv:
	python3.8 -m venv ./.churn_venv

install:
	python3.8 -m pip install -r ./requirements_py3.8.txt

venv:
	. ./.churn_venv/bin/activate

run: venv
	python3.8 ./src/churn_library.py

test: venv
	python3.8 -m pytest ./src --log-cli-level=DEBUG

lint:
	autopep8 --in-place --aggressive -r ./src && \
	pylint ./src