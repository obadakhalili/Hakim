install: requirements-dev.txt
	pip install -r requirements-dev.txt

format:
	black --exclude .venv ./actions
