import os

from openai import OpenAI

from biomaker.utils.env import load_local_env

load_local_env()
client = OpenAI()

completion = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ],
    temperature=0.0,
    max_completion_tokens=128,
)

print(completion.choices[0].message)

for model in client.models.list():
    print(model.id)
