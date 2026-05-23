import time
import random
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

def call_with_retry(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                # 控制输出长度，减少单次处理时间
                max_tokens=2048
            )
            return response
        except Exception as e:
            if "1302" in str(e) or "速率限制" in str(e):
                # 指数退避：1s, 2s, 4s, 8s, 16s...
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"触发速率限制，等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("超过最大重试次数")

# 批量调用时增加间隔
for item in data_list:
    result = call_with_retry([{"role": "user", "content": item}])
    time.sleep(0.5)  # 每次请求后至少间隔 0.5-1 秒