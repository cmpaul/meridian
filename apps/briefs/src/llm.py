import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def call_llm(
    model: str, messages: list[dict[str, str]], temperature: float = 0.0
) -> tuple[str, tuple[int, int]]:
    """
    Call a language model with the given parameters and return the response.

    Args:
        model: The name of the language model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        messages: A list of message dictionaries containing 'role' and 'content' keys
        temperature: Controls randomness in the response (0.0 = deterministic, higher values = more random)

    Returns:
        A tuple containing:
        - The generated text response as a string
        - A tuple of (prompt_tokens, completion_tokens) representing token usage

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> response, (prompt_tokens, completion_tokens) = call_llm("gpt-4", messages)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=temperature,
    )

    return response.choices[0].message.content, (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    # return {
    #     "answer": response.choices[0].message.content,
    #     "usage": {
    #         "input_tokens": response.usage.prompt_tokens,
    #         "output_tokens": response.usage.completion_tokens,
    #     },
    # }
