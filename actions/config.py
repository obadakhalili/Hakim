import os
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

env = os.environ.get("ENV", "dev")
infermedica_app_id = os.environ.get("INFERMEDICA_APP_ID", "")
infermedica_app_key = os.environ.get("INFERMEDICA_APP_KEY", "")
gcloud_project_id = os.environ.get("GCLOUD_PROJECT_ID", "")
