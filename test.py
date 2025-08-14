import requests

url = "http://chatbot-ai.koreacentral.cloudapp.azure.com/chatbot/generate"
data = {
    "session_id": "user-12",
    "message_id": "1",
    "parent_id": None,
    "user_id": "1",
    "content": "오늘 날씨는?",
    "context": ""
}
headers = {"Content-Type": "application/json; charset=UTF-8"}

response = requests.post(url, json=data, headers=headers)
print(response.text)