import argparse

from openai import OpenAI

from biomaker.utils.env import load_local_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    load_local_env()
    client = OpenAI()
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming.",
            },
        ],
        temperature=0.0,
        max_completion_tokens=128,
    )

    print(completion.choices[0].message)

    for model in client.models.list():
        print(model.id)


if __name__ == "__main__":
    main()
