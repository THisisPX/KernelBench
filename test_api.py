from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",  # 修正：去掉空格，加 /v1
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="base",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=100,  # 建议加上，防止无限生成
)

print(completion.choices[0].message.content)