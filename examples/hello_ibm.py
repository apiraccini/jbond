from dotenv import load_dotenv
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

load_dotenv()


def main():

    # Choose LLM
    model_id = "meta-llama/llama-3-1-8b-instruct"

    # Configure model parameters
    params = TextChatParameters(temperature=0.2)

    # Initialize the language model with configured settings
    model = ModelInference(
        model_id=model_id,
        credentials=Credentials(
            url=os.getenv("IBM_URL"),
            api_key=os.getenv("IBM_API_KEY"),
        ),
        project_id=os.getenv("IBM_PROJECT_ID"),
        params=params,
    )

    # Prepare chat messages
    messages = [
        {
            "role": "system",
            "content": "Your name is Jbond and you are a useful assistant.",
        },
        {"role": "user", "content": "Hi there"},
    ]

    # Get response from the model
    response = model.chat(messages=messages)
    print(response["choices"][0]["message"])


if __name__ == "__main__":
    main()
