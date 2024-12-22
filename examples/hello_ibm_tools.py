import os
import inspect
import json
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

load_dotenv()


def parse_function(func):
    """
    Parses a given function to extract its representation for LLM tool callling.
    Args:
        func (function): The function to parse.
    Returns:
        dict: A dictionary containing the parsed function information
    """
    # Get the function name and docstring
    func_name = func.__name__
    docstring = inspect.getdoc(func)

    # Extract parameters
    signature = inspect.signature(func)
    parameters = []
    for param in signature.parameters.values():
        param_info = {
            "name": param.name,
            "type": (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            ),
            "default": (
                param.default if param.default != inspect.Parameter.empty else None
            ),
        }
        parameters.append(param_info)

    # Determine required parameters
    required_params = [
        param["name"] for param in parameters if param["default"] is None
    ]

    # Construct the output dictionary
    output = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring,
            "parameters": {
                "properties": {
                    param["name"]: {"type": param["type"], "default": param["default"]}
                    for param in parameters
                },
                "required": required_params,
                "type": "object",
            },
        },
    }

    return output


def add(a: float, b: float) -> float:
    """
    Adds two numbers.

    Parameters:
    - a (float): The first number.
    - b (float): The second number.

    Returns:
    - float: The result of a + b.
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.

    Parameters:
    - a (float): The first number.
    - b (float): The second number.

    Returns:
    - float: The result of a * b.
    """
    return a * b


def main():

    # Choose LLM
    model_id = "meta-llama/llama-3-1-8b-instruct"

    # Initialize the language model with configured settings
    model = ModelInference(
        model_id=model_id,
        credentials=Credentials(
            url=os.getenv("IBM_URL"),
            api_key=os.getenv("IBM_API_KEY"),
        ),
        project_id=os.getenv("IBM_PROJECT_ID"),
    )

    # Prepare tools
    formatted_tools = [parse_function(tool) for tool in [add, multiply]]
    function_mapping_dict = {
        "add": add,
        "multiply": multiply,
    }
    print(json.dumps(formatted_tools[0], indent=4))

    # Prepare chat messages
    messages = [
        {
            "role": "system",
            "content": "Your name is Jbond and you are a useful assistant.",
        },
        {"role": "user", "content": "Hi there, what is 196*53?"},
    ]

    # Get response from the model
    response = model.chat(
        messages=messages, tools=formatted_tools, tool_choice_option="auto"
    )
    print(response["choices"][0]["message"])

    if "tool_calls" in response["choices"][0]["message"]:
        tool_call = response["choices"][0]["message"]["tool_calls"]
        function_name = tool_call[0]["function"]["name"]
        function_params = json.loads(tool_call[0]["function"]["arguments"])
        print(
            f"Executing function: `{function_name}`, with parameters: {function_params}"
        )

        function_result = function_mapping_dict[function_name](**function_params)
        print(f"Function result: {function_result}")


if __name__ == "__main__":
    main()
