import requests
import json

url = 'http://localhost:8080/'

def predict(txt):
    myjson = {'text': txt}
    urlReq = url + 'predict'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads( x.text)
    print(jsonRsp)
    print(    jsonRsp["predictions"][0]["PSC"],    jsonRsp["predictions"][0]["desc"],    jsonRsp["predictions"][0]["status"])
    return jsonRsp

def accept(txt, psc):
    myjson = {'text': txt, "PSC":psc}
    urlReq = url + 'accept'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads( x.text)
    print( jsonRsp["status"])
    return jsonRsp

print(predict("test"))
print(accept("test","41111"))
#print the response text (the content of the requested file):

