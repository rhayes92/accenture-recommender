################### User Interface ########################## 

# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import pandas as pd
import pandas.io.json._normalize
from pandas import ExcelWriter
from openpyxl.workbook import Workbook
# NLP Packages
import preprocessing as pp


### Structure and Layout ###
window = Tk()
window.title("NLP PSC")
window.geometry("1200x800")
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

# API for predict
def predict(txt):
    myjson = {'text':txt}
    urlReq = url + 'predict'
    x = requests.post(urlReq, json = myjson)
    jsonRsp = json.loads(x.text)
    # print(jsonRsp)
    # print(jsonRsp["predictions"][0]["PSC"],    jsonRsp["predictions"][0]["desc"],    jsonRsp["predictions"][0]["status"])
    return jsonRsp

# # Predict using raw text
# def predict_psc():
#     raw_text = str(TitleName.get()+" "+DesName.get())
#     prediction=predict(raw_text)
#     psc_display.insert(tk.END,prediction)

prediction=[]
# jsonPrd=[]
# Predict using preprocessed text
def predictaction():
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
    global prediction
    prediction=predict(cleantext)
    # global jsonPrd
    jsonPrd=pd.json_normalize(prediction, 'predictions')
    df_rows=jsonPrd.to_numpy().tolist()
    tv["column"]=list(jsonPrd.columns)
    tv["show"]="headings"
    for column in tv["column"]:
        tv.heading(column,text=column)
    for row in df_rows:
        tv.insert("","end",values=row)
    # print(jsonprd)
    # psclist=[prediction['predictions'][0],"\n",prediction['predictions'][1],"\n",prediction['predictions'][2],"\n",prediction['predictions'][3],"\n",prediction['predictions'][4]]

    # psc_display.insert(tk.END,jsonPrd.iloc[:,0:2])
    # psc_display.insert(INSERT,prediction['predictions'][0],"\n",prediction['predictions'][1],"\n",prediction['predictions'][2])
    psc1_display.insert(tk.END,prediction['predictions'][0]['PSC'])

# Insert the list of top recommendations to dropdown menu
def updateMenu(pscMenu,psc_option):
    global prediction
    # psclist=[prediction['predictions'][0]['PSC'],prediction['predictions'][1]['PSC'],prediction['predictions'][2]['PSC'],prediction['predictions'][3]['PSC'],prediction['predictions'][4]['PSC']]
    psclist=[prediction['predictions'][1]['PSC'],prediction['predictions'][2]['PSC'],prediction['predictions'][3]['PSC'],prediction['predictions'][4]['PSC']]
    pscMenu.configure(state='normal')
    menu=pscMenu['menu']
    menu.delete(0, 'end')
    for name in psclist:
        # Add menu items.
        menu.add_command(label=name, command=lambda name=name: psc_option.set(name))
    psc_option.set('Select...')

# API for accept psc
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

# Save the order name and description along with correct psc
def accept_psc():
    raw_text = str(TitleName.get()+" "+DesName.get())
    if pscName.get(): #If suggested psc is entered, use pscName
        updated_psc=pscName.get()
    else:
        updated_psc=psc_option.get() #No suggested psc is entered; use top5 recommendation
    acceptpsc=accept(raw_text,updated_psc)
    result_display.insert(tk.END,acceptpsc)

# API for retraining the model
def retrain():
    myjson = {}
    urlReq = url + 'retrain'
    x = requests.post(urlReq, json = myjson, timeout=14400)
    jsonRsp = json.loads( x.text)
    print( jsonRsp["status"])
    return jsonRsp

def retrain_psc():
    global bool_value2
    if bool_value2.get()==1:
        retrain()

##TRACKER

count=0 
def click0():
    global count
    count=count + 1
    label0.configure(text="Generated "+ str(count) + " recommendation(s).")

#Count the number of times 
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
    val3 =[] 
    a = psc1_display.get(1.0,tk.END+"-1c")
    b = psc_option.get()
    c = pscEn.get()
    sel_string="Select..."
    if c:
        val3=c
    elif sel_string not in b:
        # user selected a psc
        val3=b
    else: 
        val3=a
    data.append([val1, val2, val3])
    # print(data)

record=[]
def export():
    global count,count1,count2
    record.append([count,count1,count2])
    df1 = pd.DataFrame(data,columns = ["Order Title","Line Description","PSC Recommendation"])
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
    # psc_display.delete('1.0',END)
    tv.delete(*tv.get_children())
    psc1_display.delete('1.0',END)

def clear_menu(pscMenu,psc_option):
    menu=pscMenu['menu']
    menu.delete(0, 'end')
    psc_option.set('')

def clear_edited():
    pscEn.delete(0,END)

def clear_status():
    result_display.delete('1.0',END)


### Tab1 ###


label1 = Label(tab1, text= 'Product Service Code Recommendation Engine',padx=5, pady=5,font="Arial 16 bold")
label1.grid(row=1,column=0, padx=5, pady=5, columnspan = 4)


S1Lb1 = Label(tab1, text="Order Title", fg="black", bg="white",font = ('Calibri',12))
S1Lb1.grid(row=2, column=0, padx=5, pady=5, sticky=E)
TitleName = StringVar()
TitleEn = Entry(tab1, textvariable=TitleName, width=60)
TitleEn.grid(row=2, column=1, ipady=4)
clear_input1=tk.Button(tab1,text="Clear Input",command=clear_text1, activebackground = 'white',font = ('Calibri',12))
clear_input1.grid(row=2, column=2, padx=5, pady=5,sticky=W+E)


S1Lb2 = Label(tab1, text="Line Description", fg="black", bg="white",font = ('Calibri',12))
S1Lb2.grid(row=3, column=0, padx=5, pady=5,sticky=E)
DesName = StringVar()
DesEn = Entry(tab1, textvariable=DesName, width=60)
DesEn.grid(row=3, column=1, ipady=4)
clear_input2=tk.Button(tab1,text="Clear Input",command=clear_text2, activebackground = 'white',font = ('Calibri',12))
clear_input2.grid(row=3, column=2, padx=5, pady=5,sticky=W+E)


#S1Lb3 = Label(tab1, text="Preprocessed Text", fg="black", bg="white")
#S1Lb3.grid(row=5, column=0, padx=5, pady=5,sticky=W+E)
#text_display = Text(tab1,height=2, width=40)
#text_display.grid(row=5, column=1,padx=10,sticky=W+E)
#clean_but=tk.Button(tab1,text="Clean Text",command=action1, activebackground = 'white')
#clean_but.grid(row=5, column=2, padx=5, pady=5,sticky=W+E)

S1Lb3 = Label(tab1, text="PSC Status", fg="black", bg="white",font = ('Calibri',12))
S1Lb3.grid(row=5, column=0, padx=5, pady=5,sticky=E)
cols = ('PSC', 'desc', 'status')
tv = ttk.Treeview(tab1, columns=cols, show='headings')
tv.grid(row=5, column=1)


# psc_display = Text(tab1, height=12, width=50)
# psc_display.grid(row=6, column=1,padx=5,sticky=W+E)
lookupbutton=tk.Button(tab1,text="Generate Recommendation",command=lambda:[click0(),predictaction(),updateMenu(pscMenu,psc_option)], activebackground = 'white',font = ('Calibri',12)) 
lookupbutton.grid(row=5, column=2, padx=0, pady=0,sticky=W+E)

S1Lb4 = Label(tab1, text="PSC Recommendation", fg="black", bg="white",font = ('Calibri',12))
S1Lb4.grid(row=7, column=0, padx=5,pady=5,sticky=E)
psc1_display = Text(tab1, height=2, width=20)
psc1_display.grid(row=7, column=1,padx=5,sticky=W)
correctbutton=tk.Button(tab1,text="Accept Recommendation",command=lambda:[click1(),savelog()], activebackground = 'white',font = ('Calibri',12))
correctbutton.grid(row=7, column=1, padx=5,pady=5,sticky=E)
clear_rec=tk.Button(tab1,text="Clear Recommendation",command=lambda:[clear_result(),clear_menu(pscMenu,psc_option)], activebackground = 'white',font = ('Calibri',12))
clear_rec.grid(row=7, column=2, padx=5, pady=5,sticky=W+E)


S1Lb5 = Label(tab1, text="Select Other Product Service Code", fg="black", bg="white",font = ('Calibri',12))
S1Lb5.grid(row=9, column=0, padx=5, pady=5,sticky=E)
## Dropdown Version 
psc_option = StringVar()
# Dictionary with options
# psc_list = {'1','2','3','4','5'} # will have to enter actual PSC list here
psc_option.set('Select...') # set the default option
pscMenu=OptionMenu(tab1,psc_option,"Select from Other Recommendation")
pscMenu.grid(row = 9, column =1, sticky=W+E)
pscMenu.configure(bg = 'white')
# psc_option.trace('w', change_dropdown) # link function to change dropdown
# clear_edited_record=tk.Button(tab1,text="Reset PSC",command=clear_edited, activebackground = 'white', font = ('Calibri',12))
# clear_edited_record.grid(row=9, column=2, padx=5, pady=5,sticky=W+E)
## Input Version 
pscName = StringVar()
pscEn = Entry(tab1, textvariable=pscName, width=30)
pscEn.grid(row=9, column=2, ipady=4)
clear_edited_record=tk.Button(tab1,text="Clear PSC Input",command=clear_edited, activebackground = 'white',font = ('Calibri',12))
clear_edited_record.grid(row=10, column=2, padx=5, pady=5,sticky=W+E)

# Save Control
bool_value1 = IntVar()
checkbut=Checkbutton(tab1,text='Save Permission',variable=bool_value1, font = ('Calibri',12))
checkbut.grid(row=10, column=1, padx=5,pady=5,sticky=W)
acceptbutton=tk.Button(tab1,text="Select Product Service Code",command=lambda:[click2(),accept_psc(),savelog()], activebackground = 'white',font = ('Calibri',12))
acceptbutton.grid(row=10, column=1, padx=5,pady=5,sticky=E)


S1Lb7 = Label(tab1, text="Edited PSC Status", fg="black", bg="white",font = ('Calibri',12))
S1Lb7.grid(row=12, column=0, padx=5,pady=5,sticky=E)
result_display = Text(tab1,height=2, width=40)
result_display.grid(row=12, column=1,padx=10,sticky=W+E)

# Retrain Control
bool_value2 = IntVar()
retrainbut=Checkbutton(tab1,text='Retrain Permission',variable=bool_value2, font = ('Calibri',12))
retrainbut.grid(row=15, column=1, padx=5,pady=5,sticky=W)
retrainbutton=tk.Button(tab1,text="       Retrain       ",command=retrain_psc, activebackground = 'white',font = ('Calibri',12))
retrainbutton.grid(row=15, column=1, padx=5, pady=5,sticky=E)

S1Lb8 = Label(tab1, text="Retrained PSC Status", fg="black", bg="white",font = ('Calibri',12))
S1Lb8.grid(row=16, column=0, padx=5,pady=5,sticky=E)
retrain_display= Text(tab1,height=2, width=40)
retrain_display.grid(row=16, column=1,padx=10,sticky=W+E)




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

about_label = Label(window,text="NLP GUI V.2.0.1 Team Innovation Geeks",pady=5,padx=5)
about_label.pack(fill='both')


window.mainloop()
