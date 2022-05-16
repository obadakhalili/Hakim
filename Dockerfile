FROM rasa/rasa-sdk:3.1.0

WORKDIR /app

COPY actions/requirements-actions.txt ./
USER root
RUN pip install -r requirements-actions.txt
USER 1001

COPY ./actions /app/actions
COPY ./.env /app/.env
COPY ./google_service_account_credentials.json /app/google_service_account_credentials.json

ENTRYPOINT ["rasa"]
CMD ["run", "actions"]
