import requests

def chat(prompt,history):
    resp = requests.post(
        url = 'http://127.0.0.1:8000',
        json = {"prompt":prompt,"history":history},
        headers = {"content-type": "application/json;charset=UTF-8"}
    )
    return resp.json()['response'],resp.json()['history']

history = []
while True:
    response ,history = chat(input("Query:"),history)
    print('Response:',response)
