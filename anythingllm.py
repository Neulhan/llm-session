import requests

headers = {
    'accept': 'application/json',
    'Authorization': 'Bearer BWB3TB4-3TAMHEA-KRPBF02-XX63MFY',
    'Content-Type': 'application/json',
}

json_data = {
    'message': '8월 8일 일정 알려줘',
    'mode': 'query',
}

response = requests.post(
    'http://localhost:3001/api/v1/workspace/258d5c7b-7a0f-4b43-8a80-b72f04914017/chat',
    headers=headers,
    json=json_data,
)

print(response.json())
