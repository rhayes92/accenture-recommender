# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox as messagebox
import pandas as pd

### NLP Packages
import preprocessing1 as pp


class TkDemo():
    def __init__(self):
        window = Tk()
        window.title("NLP PSC")
        window.geometry("800x800")
        
        frame1 = Frame(window)
        frame1.pack(fill=X)
        ## New Data Input
        title1 = Label(frame1, text= 'PSC Code Recommendation',padx=5, pady=5,font="Verdana 10 bold")
        title1.grid(row=1,column=0)
        # Entry of title
        S1Lb1 = Label(frame1, text=" Item Title", fg="black", bg="white")
        S1Lb1.grid(row=2, column=0, padx=3, pady=3, sticky=W+E)
        self.TitleName = StringVar()
        TitleEn = Entry(frame1, textvariable=self.TitleName, width=50)
        TitleEn.grid(row=2, column=1)
        # Entry of desc
        S1Lb2 = Label(frame1, text=" Item Description", fg="black", bg="white")
        S1Lb2.grid(row=3, column=0, padx=3, pady=3,sticky=W+E)
        self.DesName = StringVar()
        DesEn = Entry(frame1, textvariable=self.DesName, width=50)
        DesEn.grid(row=3, column=1)
        # Button for lookup
        lookupbutton=tk.Button(frame1,text="Lookup")
        lookupbutton.grid(row=3, column=2, padx=5, pady=5,sticky=W+E)
        
        frame2 = Frame(window)
        frame2.pack(fill=X)
        # Display of recommendation
        S1Lb3 = Label(frame2, text=" Preprocessed Text", fg="black", bg="white")
        S1Lb3.grid(row=4, column=0, padx=3, pady=3,sticky=W+E)
        S1Lb4 = Label(frame2, text=" PSC Recommendation", fg="black", bg="white")
        S1Lb4.grid(row=5, column=0, padx=5,pady=5,sticky=W+E)
        text_display = Text(frame2,height=2, width=40)
        text_display.grid(row=4, column=1,padx=10,sticky=W+E)
        psc_display = Text(frame2,height=5, width=40)
        psc_display.grid(row=5, column=1,padx=10,sticky=W+E)
         # Button for accept



        window.mainloop()


TkDemo()
