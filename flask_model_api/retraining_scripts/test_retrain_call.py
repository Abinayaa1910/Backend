import requests

url = 'http://127.0.0.1:5000/retrain'
file_path = 'dummy_customer_dataset_4000_clean.xlsx'  # Change to your test file

with open(file_path, 'rb') as f:
    files = {'files': f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
