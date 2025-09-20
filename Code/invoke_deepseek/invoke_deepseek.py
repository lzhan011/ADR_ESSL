import requests
import json
import time

# 替换成你的 DeepSeek API Key
API_KEY = os.environ["DEEPSEEK_API_KEY"]
API_URL = 'https://api.deepseek.com/v1/chat/completions'  # 假设 DeepSeek 接口遵循 OpenAI 格式

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}



results = []  # 用于收集每个对话的记录

def send_prompt(prompt):
    payload = {
        "model": "deepseek-chat",  # 或 "deepseek-coder", "deepseek-chat-7b", 等
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API 请求失败：{response.status_code}, {response.text}")



if __name__ == '__main__':

    # 示例 prompts（你可以替换为自己的一组 prompts）
    prompts = [
        """
        You are a SAT logic solver.
Please use the DPLL algorithm to step-by-step solve the following 3-CNF formula. At each step, record:

* Which variable is assigned
* Any propagated implications
* Whether the formula is satisfied or a conflict occurs
  Finally, output:
* Number of branches (i.e., decision points)
* Number of conflicts (i.e., backtracking steps)

The formula is:

```
(x1 ∨ ¬x2)
(x2 ∨ ¬x3)
(x1 ∨ x3)
```
"""
    ]


    # 循环执行多次
    for prompt in prompts:
        try:
            print(f"发送 prompt: {prompt}")
            reply = send_prompt(prompt)
            print(f"收到 response: {reply}\n")

            # 保存到结果列表
            results.append({
                "prompt": prompt,
                "response": reply
            })

            time.sleep(1)  # 可选：防止速率限制
        except Exception as e:
            print("出错：", e)

    # 保存所有结果到 JSON 文件
    with open("deepseek_responses.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("所有prompt和response已保存到 deepseek_responses.json")
