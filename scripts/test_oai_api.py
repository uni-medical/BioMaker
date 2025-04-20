import os
from openai import OpenAI
client = OpenAI()


completion = client.chat.completions.create(
    # model="gpt-4o-mini",
    model="gpt-4o-mini",
    # model="anthropic/claude-3.7-sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)


# print(client.models.list())