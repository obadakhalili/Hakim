install: requirements.txt
	pip install -r requirements.txt

format:
	black --exclude .venv ./actions
