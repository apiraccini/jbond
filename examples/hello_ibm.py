from dotenv import load_dotenv
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

load_dotenv()

def main():
    
    # Initialize IBM credentials and project ID
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        api_key=os.getenv("IBM_API_KEY"),
    )
    project_id = os.getenv("IBM_PROJECT_ID")

    # Choose LLM
    model_id = "meta-llama/llama-3-1-8b-instruct"

    # Configure model parameters
    params = TextChatParameters(
        temperature=0.2
    )

    # Initialize the language model with configured settings
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params
    )

    # Prepare chat messages    
    messages = [
        {
            "role": "system", 
            "content": "Your name is Jbond and you are a useful assistant."
        },
        {
            "role": "user", 
            "content": "Hi there"
        }
    ]

    # Get response from the model
    response = model.chat(messages=messages)
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
