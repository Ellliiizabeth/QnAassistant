import requests

url = "http://127.0.0.1:8888/ask"
data = {
    "question": "特朗普有什么性格特征？",
    "top_k": 5,
    "debug": True
}

response = requests.post(url, json=data)
print(response.json())
