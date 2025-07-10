import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example input for iris

print("📤 Sending request to API...")

try:
    response = requests.post(url, json=data)
    print("✅ Status Code:", response.status_code)
    print("📦 Response JSON:", response.json())
except Exception as e:
    print("❌ ERROR OCCURRED:")
    print(e)
