from rasa_sdk import Action, events
from google.cloud import translate
from google.oauth2 import service_account
import infermedica_api as infermedica

from actions import config

infermedica_api = infermedica.BasicAPIv3Connector(
    app_id=config.infermedica_app_id,
    app_key=config.infermedica_app_key,
    dev_mode=config.env == "dev",
)

gtranslation_api = translate.TranslationServiceClient(
    credentials=service_account.Credentials.from_service_account_file(
        filename="./google_service_account_credentials.json"
    )
)


class ActionSessionStart(Action):
    def name(self):
        return "action_session_start"

    def run(self, dispatcher, tracker, domain):
        # TODO: set interview-id slot
        return []


class ActionSetupInterview(Action):
    def name(self):
        return "action_setup_interview"

    def run(
        self,
        dispatcher,
        tracker,
        domain,
    ):
        try:
            taken_events = []
            collected_observations = tracker.get_slot("collected_observations")
            observations_msg = None

            if tracker.get_intent_of_latest_message() == "observations_report":
                observations_msg = tracker.latest_message["text"]

                translation_response = gtranslation_api.translate_text(
                    parent=f"projects/{config.gcloud_project_id}",
                    contents=[observations_msg],
                    source_language_code="ar",
                    target_language_code="en",
                )

                observations_msg = translation_response.translations[0].translated_text

                # NOTE: consider parsing the message and validating it has observations mentions before saving it

                if len(observations_msg) > 2048:
                    dispatcher.utter_message(response="utter_observations_too_long")
                    return taken_events

                taken_events.append(
                    events.SlotSet("last_observations_report_msg", observations_msg)
                )
            else:
                observations_msg = tracker.get_slot("last_observations_report_msg")

            sex = tracker.get_slot("sex")
            if not sex:
                dispatcher.utter_message(response="utter_specify_sex")
                return taken_events

            age = tracker.get_slot("age")
            if not age:
                dispatcher.utter_message(response="utter_specify_age")
                return taken_events

            age = int(age)
            if age < 12:
                dispatcher.utter_message(response="utter_pediatrics_not_supported")
                return taken_events

            if age > 130:
                dispatcher.utter_message(response="utter_age_too_high")
                return taken_events

            msg_parsing = infermedica_api.parse(
                {
                    "text": observations_msg,
                    "age": {"value": age},
                    "sex": sex,
                    "correct_spelling": False,
                    "context": [obs["id"] for obs in collected_observations],
                }
            )
            observations_msg, was_obvious = (
                msg_parsing[attr] for attr in ["mentions", "obvious"]
            )

            if len(observations_msg) == 0:
                dispatcher.utter_message(response="utter_no_observations")
                return taken_events

            if not was_obvious:
                # NOTE: consider validating the detected observations with the user before going any further
                pass

            if len(collected_observations) == 0:
                common_options = {
                    "sex": sex,
                    "age": {"value": age},
                    "evidence": [
                        {"id": obs["id"], "choice_id": obs["choice_id"]}
                        for obs in observations_msg
                    ],
                }

                observations_suggestions = []

                for suggest_method, source in [
                    # NOTE: consider comaenting out the following suggestion methods
                    ("symptoms", "suggest"),
                    ("demographic_risk_factors", "suggest"),
                    ("evidence_based_risk_factors", "suggest"),
                    ("red_flags", "red_flags"),
                ]:
                    method_suggestions = infermedica_api.suggest(
                        {**common_options, "suggest_method": suggest_method}
                    )

                    observations_suggestions = observations_suggestions + [
                        {"id": suggestion["id"], "source": source}
                        for suggestion in method_suggestions
                    ]

                taken_events.append(
                    events.SlotSet(
                        "observations_questions",
                        observations_suggestions,
                    )
                )

            collected_observations = collected_observations + [
                {"id": obs["id"], "state": obs["choice_id"], "source": "initial"}
                for obs in observations_msg
            ]

            taken_events.append(
                events.SlotSet("collected_observations", collected_observations)
            )

            return taken_events
        except:
            pass


class ActionDiagnose(Action):
    def name(self):
        return "action_diagnose"

    def run(
        self,
        dispatcher,
        tracker,
        domain,
    ):
        try:
            observations_questions = tracker.get_slot("observations_questions")
            collected_observations = tracker.get_slot("collected_observations")
            latest_message_intent = tracker.get_intent_of_latest_message()

            taken_events = [
                events.SlotSet("observations_questions", observations_questions),
                events.SlotSet("collected_observations", collected_observations),
            ]

            if latest_message_intent in [
                "affirm",
                "deny",
                "dont_know",
            ]:
                question_to_answer = observations_questions.pop(0)
                observation = {
                    "id": question_to_answer["id"],
                    "state": {
                        "affirm": "present",
                        "deny": "absent",
                        "dont_know": "unknown",
                    }[latest_message_intent],
                    "source": question_to_answer.get("source"),
                }
                collected_observations.append(observation)

            if len(observations_questions) == 0:
                # TODO: use interview-id slot
                diagnosis = infermedica_api.diagnosis(
                    {
                        "sex": tracker.get_slot("sex"),
                        "age": {"value": tracker.get_slot("age")},
                        "evidence": [
                            {
                                "id": obs["id"],
                                "choice_id": obs["state"],
                                "source": obs.get("source"),
                            }
                            for obs in collected_observations
                        ],
                        "extras": {"disable_groups": True},
                    }
                )

                if diagnosis["should_stop"]:
                    # TODO: send a more detailed condition
                    dispatcher.utter_message(diagnosis["conditions"][0]["common_name"])
                    return taken_events

                observations_questions.append(
                    {
                        "id": diagnosis["question"]["items"][0]["id"],
                        "text": diagnosis["question"]["text"],
                    }
                )

            observation_question = observations_questions[0]

            if not hasattr(observation_question, "text"):
                details_endpoint = (
                    infermedica_api.symptom_details
                    if observation_question["id"].startswith("s")
                    else infermedica_api.risk_factor_details
                )

                observation_details = details_endpoint(
                    observation_question["id"],
                    {
                        "age.value": tracker.get_slot("age"),
                        "sex": tracker.get_slot("sex"),
                    },
                )

                observation_question["text"] = observation_details["question"]

            translation_response = gtranslation_api.translate_text(
                parent=f"projects/{config.gcloud_project_id}",
                contents=[observation_question["text"]],
                source_language_code="en",
                target_language_code="ar",
            )

            observation_question["text"] = translation_response.translations[
                0
            ].translated_text

            dispatcher.utter_message(observation_question["text"])

            return taken_events
        except:
            pass
