freeze:
	touch requirements.txt && pip freeze > requirements.txt

install: requirements.txt
	pip install -r requirements.txt

format:
	black --exclude .venv ./actions
