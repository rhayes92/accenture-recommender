# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import pandas as pd
### NLP Packages
import preprocessing1 as pp
# import cnn as c

### Structure and Layout ###
window = Tk()
window.title("NLP PSC")
window.geometry("800x500")
window.config(background='black')

# TAB LAYOUT
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
#tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text='PSC Recommendation')
#tab_control.add(tab2, text='Data')
tab_control.add(tab3, text='Accuracy level')

### Functions and Commands

import requests
import json

url = 'http://localhost:8080/'

def predict(txt):
    myjson = {'text': txt}
    urlReq = url + 'predict'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads( x.text)
    print(jsonRsp)
    # print(jsonRsp["predictions"][0]["PSC"],    jsonRsp["predictions"][0]["desc"],    jsonRsp["predictions"][0]["status"])
    return jsonRsp

def predict_psc():
    raw_text = str(TitleName.get()+" "+DesName.get())
    prediction=predict(raw_text)
    psc_display.insert(tk.END,prediction)

def accept(txt, psc):
    myjson = {'text': txt, "PSC":psc}
    urlReq = url + 'accept'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads( x.text)
    print( jsonRsp["status"])
    return jsonRsp

def accept_psc():
    raw_text = str(TitleName.get()+" "+DesName.get())
    updated_psc=str(pscName.get())
    acceptpsc=accept(raw_text,updated_psc)
    result_display.insert(tk.END,acceptpsc)

    # ttl = str(TitleName.get())
    # des = str(DesName.get())
    # DF=pd.DataFrame(columns=['Item Title','Item Description'])
    # DF.loc[0,'Item Title']=ttl
    # DF.loc[0,'Item Description']=des
    # DF['Item Title']=DF['Item Title'].apply(lambda x : pp.clean_text(x))
    # DF['Item Description']=DF['Item Description'].apply(lambda x : pp.clean_text(x))
    # DF['Item Title']=DF['Item Title'].apply(lambda x : pp.return_sentences(x))
    # DF['Item Description']=DF['Item Description'].apply(lambda x : pp.return_sentences(x))
    # DF['clean_title_desc'] =DF['Item Title']+" "+DF['Item Description']
    # DF['clean_title_desc'] =DF['clean_title_desc'].replace(r'\b\w{1,3}\b', "", regex=True)
    # DB=DF
    # psc_input=DB['clean_title_desc'][0]
    # psc_prediction=predict(psc_input)
    # psc_display.insert(tk.END,psc_prediction)

# import pandas as pd
# DF=pd.DataFrame()
# Get preprocessed text
def action1():
    #global DB
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
    text_display.insert(tk.END,DB['clean_title_desc'][0])


# Clear Entry & Display
def clear_text():
    TitleEn.delete(0,END)
    DesEn.delete(0,END)

def clear_result():
    text_display.delete('1.0',END)
    psc_display.delete('1.0',END)

def clear_edited():
    pscEn.delete(0,END)
    result_display.delete('1.0',END)

### Tab1

# Entry and Display columns

label1 = Label(tab1, text= 'PSC Code Recommendation',padx=5, pady=5,font="Verdana 10 bold")
label1.grid(row=1,column=0)

S1Lb1 = Label(tab1, text=" Item Title", fg="black", bg="white")
S1Lb1.grid(row=2, column=0, padx=3, pady=3, sticky=W+E)
TitleName = StringVar()
TitleEn = Entry(tab1, textvariable=TitleName, width=50)
TitleEn.grid(row=2, column=1)

S1Lb2 = Label(tab1, text=" Item Description", fg="black", bg="white")
S1Lb2.grid(row=3, column=0, padx=3, pady=3,sticky=W+E)
DesName = StringVar()
DesEn = Entry(tab1, textvariable=DesName, width=50)
DesEn.grid(row=3, column=1)

S1Lb3 = Label(tab1, text=" Preprocessed Text", fg="black", bg="white")
S1Lb3.grid(row=5, column=0, padx=3, pady=3,sticky=W+E)
text_display = Text(tab1,height=2, width=40)
text_display.grid(row=5, column=1,padx=10,sticky=W)

S1Lb4 = Label(tab1, text=" PSC Recommendation", fg="black", bg="white")
S1Lb4.grid(row=6, column=0, padx=5,pady=5,sticky=W+E)
psc_display = Text(tab1,height=5, width=40)
psc_display.grid(row=6, column=1,padx=10,sticky=W)

S1Lb5 = Label(tab1, text=" Edit PSC Here", fg="black", bg="white")
S1Lb5.grid(row=9, column=0, padx=3, pady=3,sticky=W+E)
pscName = StringVar()
pscEn = Entry(tab1, textvariable=pscName, width=50)
pscEn.grid(row=9, column=1)


# Display Results 

S1Lb6 = Label(tab1, text=" Edited result", fg="black", bg="white")
S1Lb6.grid(row=10, column=0, padx=5,pady=5,sticky=W+E)
result_display = Text(tab1,height=2, width=40)
result_display.grid(row=10, column=1,padx=10,sticky=W)



# Tab3


label0 = Label(tab3, text= 'How many lookup times',padx=5, pady=5)
def clickOK():
    global count
    count=count + 1
    label0.configure(text="Looked up for "+ str(count) + " time(s)")
count=0 #TRACKER
label0.grid(row=1, column=0)

label3 = Label(tab3, text= 'Accuracy level',padx=5, pady=5,font="Verdana 10 bold")
label3.grid(row=2, column=0)

S3Lb1 = Label(tab3, text=" Correct", fg="black", bg="white")
S3Lb1.grid(row=6, column=0, pady=5,sticky=W+E)

S3Lb3 = Label(tab3, text=" Not Correct", fg="black", bg="white")
S3Lb3.grid(column=0,row=7, pady=5,sticky=W+E)

S3Lb2 = Label(tab3, text= 'How many correct records',padx=5, pady=5)
S3Lb2.grid(column=1, row=6)
S3Lb4 = Label(tab3, text= 'How many incorrect records',padx=5, pady=5)
S3Lb4.grid(column=1, row=7)

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


tab_control.pack(expand=1, fill='both')

### BUTTON ###
clean_but=tk.Button(tab1,text=" Clean Text",command=action1)
clean_but.grid(row=5, column=2, padx=5, pady=5,sticky=W+E)

lookupbutton=tk.Button(tab1,text="Lookup",command=lambda:[clickOK(),predict_psc()])
lookupbutton.grid(row=6, column=2, padx=5, pady=5,sticky=W+E)
correctbutton=tk.Button(tab1,text="Correct",command=click1)
correctbutton.grid(row=7, column=1, padx=5,pady=5,sticky=E)
acceptbutton=tk.Button(tab1,text="Accept",command=lambda:[click2(),accept_psc()])
acceptbutton.grid(row=9, column=2, padx=5,pady=5,sticky=E)
clear_input=tk.Button(tab1,text=" Clear Input",command=clear_text)
clear_input.grid(row=7, column=2, padx=5, pady=5,sticky=W+S)
clear_result=tk.Button(tab1,text="Clear Result",command=clear_result)
clear_result.grid(row=8, column=2, padx=5, pady=5,sticky=W+S)
clear_edited_record=tk.Button(tab1,text=" Clear Input",command=clear_edited)
clear_edited_record.grid(row=10, column=2, padx=5, pady=5,sticky=W+S)






about_label = Label(window,text="NLP GUI V.0.0.1 Team Innovation Geek",pady=5,padx=5)
about_label.pack(fill='both')


window.mainloop()
