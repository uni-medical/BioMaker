import os

from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    # model="gpt-4o-mini",
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ],
    temperature=0.0,
    max_tokens=128,
)

print(completion.choices[0].message)

for model in client.models.list():
    print(model.id)
