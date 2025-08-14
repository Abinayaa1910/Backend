import requests

data = {
    "loyalty_tier": "Silver",
    "gender": "Male",
    "location": "Choa Chu Kang",
    "join_year": 2023,
    "join_month": 11,
    "join_quarter": 4
}

response = requests.post("http://127.0.0.1:5000/generate-promo", json=data)
print(response.json())
