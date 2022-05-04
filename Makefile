freeze:
	touch requirements.txt && pip-chill > requirements.txt

install: requirements.txt
	pip install -r requirements.txt

format:
	black --exclude .venv ./actions
