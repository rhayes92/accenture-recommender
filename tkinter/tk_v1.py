################### VERSION 1 - Without Dropdown PSC List ########################## 

# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import pandas as pd
from pandas import ExcelWriter
from openpyxl.workbook import Workbook
### NLP Packages
import preprocessing1 as pp


### Structure and Layout ###
window = Tk()
window.title("NLP PSC")
window.geometry("1100x700")
window.configure(bg='dark grey')

# TAB LAYOUT
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
#tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text='PSC Recommendation')
#tab_control.add(tab2, text='Data')
tab_control.add(tab3, text='Correction Log')

### Functions and Commands ###

import requests
import json

url = 'http://localhost:8080/'
# url = 'http://3.234.211.151:8080/'

# # Get preprocessed text
# def action1():
#     #global DB
#     ttl = str(TitleName.get())
#     des = str(DesName.get())
#     DF=pd.DataFrame(columns=['Item Title','Item Description'])
#     DF.loc[0,'Item Title']=ttl
#     DF.loc[0,'Item Description']=des
#     DF['Item Title']=DF['Item Title'].apply(lambda x : pp.clean_text(x))
#     DF['Item Description']=DF['Item Description'].apply(lambda x : pp.clean_text(x))
#     DF['Item Title']=DF['Item Title'].apply(lambda x : pp.return_sentences(x))
#     DF['Item Description']=DF['Item Description'].apply(lambda x : pp.return_sentences(x))
#     DF['clean_title_desc'] =DF['Item Title']+" "+DF['Item Description']
#     DF['clean_title_desc'] =DF['clean_title_desc'].replace(r'\b\w{1,3}\b', "", regex=True)
#     DB=DF
#     return DB['clean_title_desc'][0]
#     # text_display.insert(tk.END,DB['clean_title_desc'][0])

def predict(txt):
    myjson = {'text':txt}
    urlReq = url + 'predict'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads(x.text)
    print(jsonRsp)
    # print(jsonRsp["predictions"][0]["PSC"],    jsonRsp["predictions"][0]["desc"],    jsonRsp["predictions"][0]["status"])
    return jsonRsp

# Predict using raw text
# def predict_psc():
#     raw_text = str(TitleName.get()+" "+DesName.get())
#     prediction=predict(raw_text)
#     psc_display.insert(tk.END,prediction)


# Predict using preprocessed text
def predictaction1():
    ttl = str(TitleName.get())
    des = str(DesName.get())
    DF=pd.DataFrame(columns=['Item Title','Item Description'])
    DF.loc[0,'Item Title']=ttl
    DF.loc[0,'Item Description']=des
    DF['Item Title']=DF['Item Title'].apply(lambda x : pp.clean_text(x))
    DF['Item Description']=DF['Item Description'].apply(lambda x : pp.clean_text(x))
    DF['Item Title']=DF['Item Title'].apply(lambda x : pp.return_sentences(x))
    DF['Item Description']=DF['Item Description'].apply(lambda x : pp.return_sentences(x))
    DF['clean_title_desc'] =DF['Item Title']+" "+DF['Item Description']
    DF['clean_title_desc'] =DF['clean_title_desc'].replace(r'\b\w{1,3}\b', "", regex=True)
    DB=DF
    cleantext=DB['clean_title_desc'][0]
    prediction=predict(cleantext)
    # print(prediction)
    psc_display.insert(tk.END,prediction['predictions'][0]['PSC']) #top recommendation


def accept(txt, psc):
    # save set to True will save
    global bool_value1
    if bool_value1.get()==1:
        savedata=True
    else:
        savedata=False
    psc = psc.strip()
    txt = txt.strip()
    myjson = {'text':txt, "PSC":psc, 'save':savedata}
    urlReq = url + 'accept'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads( x.text)
    print( jsonRsp["status"])
    return jsonRsp

# def accept(txt, psc):
#     #save set to True will save
#     myjson = {'text': txt, "PSC":psc, 'save':True}
#     urlReq = url + 'accept'
#     x = requests.post(urlReq, json = myjson)
#     jsonRsp = json.loads( x.text)
#     print( jsonRsp["status"])
#     return jsonRsp

def accept_psc():
    raw_text = str(TitleName.get()+" "+DesName.get())
    updated_psc=str(pscName.get())
    # updated_psc=str(psc_display.get(1.0,END))
    acceptpsc=accept(raw_text,updated_psc)
    result_display.insert(tk.END,acceptpsc)

def retrain():
    myjson = {}
    urlReq = url + 'retrain'
    x = requests.post(urlReq, json = myjson, timeout=14400)
    jsonRsp = json.loads( x.text)
    print( jsonRsp["status"])
    return jsonRsp

#TRACKER
count=0 
def clickOK():
    global count
    count=count + 1
    label0.configure(text="Generated "+ str(count) + " recommendation(s).")

count1=0
def click1():
    global count1
    count1=count1+1
    S3Lb2.configure(text=str(count1) + " time(s)")

count2=0
def click2():
    global count2
    count2=count2 + 1
    S3Lb4.configure(text=str(count2) + " time(s)")

data=[]
def savelog():
    val1 = TitleEn.get()
    val2 = DesEn.get()
    val3 = psc_display.get(1.0,tk.END+"-1c")
    val4 = pscEn.get()
    data.append([val1, val2, val3, val4])
    # print(data)

record=[]
def export():
    global count,count1,count2
    # global count1
    # global count2
    record.append([count,count1,count2])
    df1 = pd.DataFrame(data,columns = ["Order Title","Line Description","PSC Recommendation","Select Other PSC"])
    df2 = pd.DataFrame(record,columns = ["Total PSC Generated","Correct","Not Correct"])
    # df1.to_csv('output.csv',index=False)
    with pd.ExcelWriter('output.xlsx') as writer:  
        df1.to_excel(writer, sheet_name='corr_log_1')
        df2.to_excel(writer, sheet_name='corr_log_2')


# Clear Entry & Display
def clear_text1():
    TitleEn.delete(0,END)

def clear_text2():
    DesEn.delete(0,END)
    
def clear_result():
    # text_display.delete('1.0',END)
    psc_display.delete('1.0',END)

def clear_edited():
    pscEn.delete(0,END)
    result_display.delete('1.0',END)

### Tab1 ###

# Entry and Display columns

label1 = Label(tab1, text= 'Product Service Code Recommendation Engine',padx=5, pady=5,font="Arial 16 bold")
label1.grid(row=1,column=0, padx=5, pady=5, columnspan = 4)

S1Lb1 = Label(tab1, text="Order Title", fg="black", bg="white",font = ('Calibri',12))
S1Lb1.grid(row=2, column=0, padx=5, pady=5, sticky=W+E)
TitleName = StringVar()
TitleEn = Entry(tab1, textvariable=TitleName, width=55)
TitleEn.grid(row=2, column=1, ipady=4)
clear_input1=tk.Button(tab1,text="Clear Input",command=clear_text1, activebackground = 'white',font = ('Calibri',12))
clear_input1.grid(row=2, column=2, padx=5, pady=5,sticky=W+E)


S1Lb2 = Label(tab1, text="Line Description", fg="black", bg="white",font = ('Calibri',12))
S1Lb2.grid(row=3, column=0, padx=5, pady=5,sticky=W+E)
DesName = StringVar()
DesEn = Entry(tab1, textvariable=DesName, width=55)
DesEn.grid(row=3, column=1, ipady=4)
clear_input2=tk.Button(tab1,text="Clear Input",command=clear_text2, activebackground = 'white',font = ('Calibri',12))
clear_input2.grid(row=3, column=2, padx=5, pady=5,sticky=W+E)


#S1Lb3 = Label(tab1, text="Preprocessed Text", fg="black", bg="white")
#S1Lb3.grid(row=5, column=0, padx=5, pady=5,sticky=W+E)
#text_display = Text(tab1,height=2, width=40)
#text_display.grid(row=5, column=1,padx=10,sticky=W+E)


S1Lb4 = Label(tab1, text="PSC Recommendation", fg="black", bg="white",font = ('Calibri',12))
S1Lb4.grid(row=6, column=0, padx=5,pady=5,sticky=W+E)
psc_display = Text(tab1, height=4, width=40)
psc_display.grid(row=6, column=1,padx=5,sticky=W+E)
lookupbutton=tk.Button(tab1,text="Generate Recommendation",command=lambda:[clickOK(),predictaction1()], activebackground = 'white',font = ('Calibri',12)) 
# lookupbutton=tk.Button(tab1,text="Generate Recommendation",command=lambda:[clickOK(),predictaction1()], activebackground = 'white',font = ('Calibri',12)) 
lookupbutton.grid(row=6, column=2, padx=0, pady=0,sticky=W+E)


correctbutton=tk.Button(tab1,text="Accept Recommendation",command=lambda:[click1(),savelog()], activebackground = 'white',font = ('Calibri',12))
correctbutton.grid(row=7, column=1, padx=5,pady=5,sticky=W+E)
clear_result=tk.Button(tab1,text="Clear Recommendation",command=clear_result, activebackground = 'white',font = ('Calibri',12))
clear_result.grid(row=7, column=2, padx=5, pady=5,sticky=W+E)


S1Lb5 = Label(tab1, text="Select Other Product Service Code", fg="black", bg="white",font = ('Calibri',12))
S1Lb5.grid(row=9, column=0, padx=5, pady=5,sticky=W+E)
pscName = StringVar()
pscEn = Entry(tab1, textvariable=pscName, width=55)
pscEn.grid(row=9, column=1, ipady=4)
clear_edited_record=tk.Button(tab1,text="Clear PSC Input",command=clear_edited, activebackground = 'white',font = ('Calibri',12))
clear_edited_record.grid(row=9, column=2, padx=5, pady=5,sticky=W+E)


bool_value1 = IntVar()
checkbut=Checkbutton(tab1,text='Save Permission',variable=bool_value1, font = ('Calibri',12))
checkbut.grid(row=10, column=1, padx=5,pady=5,sticky=W)
acceptbutton=tk.Button(tab1,text="Select Product Service Code",command=lambda:[click2(),accept_psc(),savelog()], activebackground = 'white',font = ('Calibri',12))
acceptbutton.grid(row=10, column=1, padx=5,pady=5,sticky=E)


S1Lb7 = Label(tab1, text="Edited PSC Status", fg="black", bg="white",font = ('Calibri',12))
S1Lb7.grid(row=12, column=0, padx=5,pady=5,sticky=W+E)
result_display = Text(tab1,height=2, width=40)
result_display.grid(row=12, column=1,padx=10,sticky=W+E)


S1Lb8 = Label(tab1, text="Retrained PSC Status", fg="black", bg="white",font = ('Calibri',12))
S1Lb8.grid(row=15, column=0, padx=5,pady=5,sticky=W+E)
retrain_display= Text(tab1,height=2, width=40)
retrain_display.grid(row=15, column=1,padx=10,sticky=W)
retrainbutton=tk.Button(tab1,text="Retrain",command=retrain, activebackground = 'white',font = ('Calibri',12))
retrainbutton.grid(row=15, column=2, padx=5, pady=5,sticky=W+E)


### Tab3 ###

label0 = Label(tab3, text= 'Number of product service code recommendations generated.',padx=5, pady=5)
label0.grid(row=1, column=0)

label3 = Label(tab3, text= 'Recommendation Accuracy Level',padx=5, pady=5,font="Verdana 10 bold")
label3.grid(row=2, column=0)

S3Lb1 = Label(tab3, text="Correct", fg="black", bg="white")
S3Lb1.grid(row=6, column=0, pady=5,sticky=W+E)
S3Lb2 = Label(tab3, text= 'Number of accepted recommendations.',padx=5, pady=5)
S3Lb2.grid(row=6,column=1)

S3Lb3 = Label(tab3, text="Not Correct", fg="black", bg="white")
S3Lb3.grid(row=7, column=0, pady=5,sticky=W+E)
S3Lb4 = Label(tab3, text= 'Number of rejected recommendations.',padx=5, pady=5)
S3Lb4.grid(row=7,column=1)

exportlog=tk.Button(tab3, text="Export", command=export, activebackground = 'white', font = ('Calibri',10))
exportlog.grid(row=8, column=0, padx=5, pady=5, sticky='wes')




tab_control.pack(expand=1, fill='both')

about_label = Label(window,text="NLP GUI V.1.0.1 Team Innovation Geeks",pady=5,padx=5)
about_label.pack(fill='both')


window.mainloop()
