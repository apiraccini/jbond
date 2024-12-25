import os
import inspect
import json
from dotenv import load_dotenv
from groq import Groq

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
    model_id = "llama-3.3-70b-versatile"
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Prepare tools
    formatted_tools = [parse_function(tool) for tool in [add, multiply]]
    function_mapping_dict = {
        "add": add,
        "multiply": multiply,
    }
    #print(json.dumps(formatted_tools[0], indent=4))

    # Prepare chat messages
    messages = [
        {
            "role": "system",
            "content": "Your name is Jbond and you are a useful assistant.",
        },
        {"role": "user", "content": "Hi there, what is 196*53?"},
    ]

    # Get response from the model
    response = client.chat.completions.create(
        model=model_id, messages=messages, tools=formatted_tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = function_mapping_dict[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)

            messages.append(
                {
                    "role": f"tool",
                    #"tool_name": f"{function_name}",
                    "tool_call_id": tool_call.id,
                    "content": f"{function_name}({function_args}) = {function_response}",
                }
            )

        # Make the final request with tool call results
        final_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            #tools=formatted_tools,
            #tool_choice="auto",
            max_tokens=4096,
        )

        messages.append({
            "role": "assistant",
            "content": final_response.choices[0].message.content}
        )

        print('Messages:', json.dumps(messages, indent=4))

if __name__ == "__main__":
    main()
