initvenv:
	python3.8 -m venv ./.churn_venv

venv:
	. ./.churn_venv/bin/activate

run: venv
	python ./src/churn_library.py

test: venv
	pytest ./src

lint:
	autopep8 --in-place --aggressive -r ./src && \
	pylint ./src