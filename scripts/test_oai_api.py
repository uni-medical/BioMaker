import os
from openai import OpenAI

os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = os.environ["SILICON_API_KEY"]

client = OpenAI()

# for model in client.models.list():
#     print(model.id)
# exit()

completion = client.chat.completions.create(
    # model="gpt-4o-mini",
    # model="claude-3-5-haiku-20241022",
    # model="gpt-4.1",
    # model="gpt-4o-2024-08-06",
    # model="deepseek-r1",
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ],
    temperature=0.0,
    max_tokens=128,
)

print(completion.choices[0].message)
