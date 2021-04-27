from tkinter import *
import numpy as np
import tkinter.ttk as ttk
import matplotlib.patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
import ML_Data_Prep as dp
import Zone19_model_def as zml




class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("AIS/ADS-B ML Models")
        self.geometry("700x300")
        self.topic = Label(self,text="Prepare Daily Ship CSV File and Classify using ML Model")
        self.topic.pack(side="top")
        self.topic2 = Label(self,text="")
        self.topic_data = Label(self,text="Data Preparation")
        self.topic_data.place(relx= 0.1, rely= 0.2)
        self.topic2.pack(side="top")
        #Buttons
        self.topic_ML = Label(self,text="ML Model")
        self.topic_ML.place(relx= 0.73, rely= 0.2)
        self.button_1 = Button(self, text="AIS Model",command= self.AISML)
        self.button_1.place(relx= 0.71, rely= 0.3)
        self.topic3 = Label(self,text="")
        self.topic3.pack(side="top")
        self.button_4 = Button(self, text="Select Ship File",command=self.browseFiles)
        self.button_4.place(relx= 0.09, rely= 0.3)
        #Labels for Data prep
        self.label_status = Label(self,text="Status:")
        self.label_status.place(relx= 0.01, rely= 0.5)
        self.label_count = Label(self,text="Count:")
        self.label_count.place(relx= 0.01, rely= 0.7)
        self.label_type = Label(self,text="Types?:")
        self.label_type.place(relx= 0.01, rely= 0.9)
        #Text for data prep
        self.text_status = Text(self, height = 1, width = 10)
        self.text_status.place(relx= 0.1, rely= 0.5)
        self.text_count = Text(self, height = 1, width = 10)
        self.text_count.place(relx= 0.1, rely= 0.7)
        self.text_type = Text(self, height = 1, width = 10)
        self.text_type.place(relx= 0.1, rely= 0.9)
        #Labels for ML
        self.label_sus = Label(self,text="Suspicious Ships?:")
        self.label_sus.place(relx= 0.52, rely= 0.7)
        self.text_sus = Text(self, height = 1, width = 10)
        self.text_sus.place(relx= 0.7, rely= 0.7)
        self.label_plot = Label(self,text="Plot MMSI:")
        self.label_plot.place(relx= 0.52, rely= 0.9)
        self.entry_combo = ttk.Combobox(self, width=15)
        self.entry_combo.place(relx= 0.66, rely= 0.9)
        self.entry_combo['values'] = ['cool', 'bro']
        #Buttons for inputs to ML
        self.button_inputs = Button(self, text="Inputs",command= self.inputFiles)
        self.button_inputs.place(relx= 0.62, rely= 0.5)
        self.button_types = Button(self, text="Types",command= self.typeFiles)
        self.button_types.place(relx= 0.72, rely= 0.5)
        self.button_MMSI = Button(self, text="MMSIs",command= self.mmsiFiles)
        self.button_MMSI.place(relx= 0.82, rely= 0.5)
        #Button for plot
        self.button_plot = Button(self, text="Plot",command= self.plotPrep)
        self.button_plot.place(relx= 0.9, rely= 0.9)
        
    
    def AISML(self):
        mmsi_list,self.pie_data,sus_mmsi, self.results = zml.Z19_MlModel(self.file_path1,self.file_path2,self.file_path3)
        if all(x is None for x in sus_mmsi):
            self.text_sus.insert(END,'None!')
        else:
            self.text_sus.insert(END,'Yes!')
        self.mmsi_tuple = tuple(mmsi_list)
        self.entry_combo['values'] = self.mmsi_tuple
        self.entry_combo.current(0)
        #print(self.entry_combo.get())
        
    
    def browseFiles(self):
        filename = filedialog.askopenfilename(initialdir = "C:/", title = "Select a File")
        check, count = dp.Data_prep(filename)
        self.text_status.insert(END, 'Done!')
        self.text_count.insert(END, str(count))
        self.text_type.insert(END, str(check))
        
    def inputFiles(self):
        self.file_path1 = filedialog.askopenfilename(initialdir = "C:/", title = "Select a File")
        
    def mmsiFiles(self):
        self.file_path2 = filedialog.askopenfilename(initialdir = "C:/", title = "Select a File")
        
    def typeFiles(self):
        self.file_path3 = filedialog.askopenfilename(initialdir = "C:/", title = "Select a File")
        
    def plotPrep(self):
        global pie_indx,ship_indx
        plot_input = self.entry_combo.get()
        tuple_indx = self.mmsi_tuple.index(plot_input)
        pie_indx = self.pie_data[tuple_indx]
        ship_indx = self.results[tuple_indx]
        example_plot = MlPlot()
        example_plot


class MlPlot(Tk):
    def __init__(self):
        super(MlPlot,self).__init__()
        self.title('Pie Plot')
        self.geometry('720x720')
        fig = Figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ship_types = ['Cargo','Fishing','Tug Tow', 'Other', 'Passenger','Pleasure/Sailing','Tanker']
        final_pie = pie_indx *100
        ax.pie(final_pie)
        ax.legend(ship_types)
        rounded_result = str(round(final_pie[ship_indx],2))
        fig.suptitle('Ship is predicted to be a/an {} vessel with a confidence of {}%'.format(ship_types[ship_indx],rounded_result))
        
        circle=matplotlib.patches.Circle( (0,0),0.7, color='white')
        ax.add_artist(circle)
        
        self.canvas = FigureCanvasTkAgg(fig,self)
        self.canvas.get_tk_widget().pack(side="bottom", fill="both",expand=True)
        self.canvas._tkcanvas.pack(side="top",fill="both",expand=True)




root = Root()
root.mainloop()