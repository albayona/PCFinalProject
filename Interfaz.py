
######################################
'''
        "Diferentes funciones con tkinter!"

Programación Científica
Autor: Julio Nicolás Reyes
'''
######################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk ,Image

from matplotlib import style
#plt.style.available     #Para saber qué tipos de styles existen

'''Configuramos la Ventana'''
window = tk.Tk()                # Definimos la ventana con nombre window
window.geometry('900x700')      # Tamaño de la ventana
window.title('Diferentes funciones - Programación Científica')
window.config(cursor="watch")
# tipo de cursor: "arrow","circle","clock","cross",
                                    # "dotbox","exchange","fleur","heart","man","mouse",
                                    #"pirate","plus","shuttle","sizing","spider","spraycan",
                                    # "star","target","tcross","trek","watch"


'''Definimos los Frames y sus configuraciones para organizar la GUI'''
frame1 = tk.Frame(master=window)
frame1.place(x=25, y=50)
frame1.config(bg="#F4D03F", width=300, height=140, relief=tk.GROOVE, bd=8)

frame2 = tk.Frame(master=window).pack()

framefun = tk.Frame(master=window)
framefun.place(x=25, y=470)
framefun.config(bg="#A9CCE3", width=300, height=150, relief=tk.RIDGE, bd=8)


'''Se definen todas las funciones requeridas'''
def CerrarAplicacion():
    MsgBox = tk.messagebox.askquestion ('Cerrar Aplicación','¿Está seguro que desea cerrar la aplicación?', icon = 'warning')
    if MsgBox == 'yes':
       window.destroy()
    else:
        tk.messagebox.showinfo('Retornar','Será retornado a la aplicación')


Style = ttk.Style()       #Objeto para crear estilos  # https://kite.com/python/docs/ttk.Style
Style.configure('1.TButton', font=('Comic Sans MS', 17, 'bold italic'), foreground = '#7FB3D5', background='yellow', padding=0)
Style.configure('2.TButton', font =('Times', 20, 'bold', 'underline'), foreground = '#E88A18')
'''
Para font:      (https://effbot.org/tkinterbook/tkinter-widget-styling.htm)
    family − The font family name as a string.
    size − The font height as an integer in points. To get a font n pixels high, use -n.
    weight − "bold" for boldface, "normal" for regular weight.
    slant − "italic" for italic, "roman" for unslanted.
    underline − 1 for underlined text, 0 for normal.
    overstrike − 1 for overstruck text, 0 for normal.
'''

# Las diferentes opciones están dadas por: (states,value)
Style.map("1.TButton",
          foreground=[('pressed', 'orange'), ('active', '#1A5276')],
          background=[('pressed', '!disabled', 'yellow'), ('active', 'yellow')])

Style.map("2.TButton",
           foreground=[('pressed', 'yellow'), ('active', '#34495E')],
           background=[('pressed', '!disabled', 'black'), ('active', 'white')])


Boton2 = ttk.Button(master=window, text="Cerrar", style="2.TButton", command = CerrarAplicacion).place(x=750,y=650)
#messagebox.askokcancel("Title","The application will be closed")

'''
Para ttk Button         (https://www.geeksforgeeks.org/python-add-style-to-tkinter-button/)
    command: A function to be called when button is pressed.
    text: Text which appears on the Button.
    image: Image to be apper on the Button.
    style: Style to be used in rendering this button.
'''


'''Ecuación'''
lbl_titulo = tk.Label(master=frame1, bg="#F4D03F", font=('Comic Sans MS', 15, 'bold italic'), text=f"Ecuación: y = 5x + 5").place(x=70,y=10)
lbl_x = tk.Label(master=frame1,
                  font=('math', 15, 'bold italic'),
                  bg="#F4D03F",
                  text="x =",
                  #relief = tk.RIDGE,
                  borderwidth = 3,
                  width = 8).place(x=70, y=40)
def obtener():
    x = int(valX.get())
    y = 5*x + 5
    resul.set(y)

valX = tk.StringVar()
ValorX = tk.Spinbox(master=frame1, from_=0, to=100, textvariable = valX, width = 5).place(x=150, y=40)
BotonCalcular = ttk.Button(master=frame1, text="y =", style="1.TButton", command = obtener, width = 3).place(x=70, y=80)

resul = tk.StringVar()
labelX = tk.Label(master=frame1, textvariable = resul, width = 7, bg = '#B2BABB', fg='blue').place(x=150, y=80)



'''Para el cálculo del conversor de unidades'''
def celsius_to_farenheit():
    celsius = ValorCelsius.get()
    fahrenheit = ((9/5) * float(celsius)) + 32
    var2.set(f"{round(fahrenheit, 3)} \N{DEGREE FAHRENHEIT}")

ValorCelsius = tk.StringVar()
ent_temperature = tk.Entry(master=frame2, textvariable=ValorCelsius, width=5).place(x=70,y=300)
lbl_temp = tk.Label(master=frame2, text="\N{DEGREE CELSIUS}").place(x=140,y=300)

btn_convert = tk.Button(
    master = frame2,
    text = "\N{RIGHTWARDS BLACK ARROW}",
    command = celsius_to_farenheit).place(x=170, y=300)

var2 = tk.StringVar()
lbl_result = tk.Label(master=frame2, textvariable=var2, text="\N{DEGREE FAHRENHEIT}").place(x=200, y=300)




'''Graficador de funciones'''
def fun(t):
    if opcion.get() == 1:
        return np.sin(t)
    elif opcion.get() == 2:
        return np.cos(t)
    elif opcion.get() == 3:
        return np.exp(t)
    elif opcion.get() == 4:
        return np.log(t)
    elif opcion.get() == 5:
        return np.sqrt(t)


def grafica():
    fig = plt.Figure(figsize=(4, 2), dpi=100)
    t = np.arange(0,10, 0.01)
    fig.add_subplot(111).plot(t, fun(t))     # subplot(filas, columnas, item)
    fig.suptitle(opcion.get())

    plt.close()
    plt.style.use('seaborn-darkgrid')
    Plot = FigureCanvasTkAgg(fig, master=window)
    Plot.draw()

    #toolbar = NavigationToolbar2Tk(Plot, window)
    #toolbar.update()
    Plot.get_tk_widget().place(x=350,y=50)

    #def on_key_press(event):
    #    print("you pressed {}".format(event.key))
    #    key_press_handler(event, Plot, toolbar)
    #Plot.mpl_connect("key_press_event", on_key_press)


lbl_gra = tk.Label(master=framefun, bg="#C8FB63", font=('Comic Sans MS', 15, 'bold italic'), text=f"Selecciona la función").place(x=70,y=10)

opcion = tk.IntVar()
Nombre = tk.StringVar()
seno = tk.Radiobutton(master=framefun, text='sen(x)', value=1, command=grafica, variable=opcion, bg='#A9CCE3',font=('math', 15, 'bold italic')).place(x=30,y=50)
coseno = tk.Radiobutton(master=framefun, text='cos(x)', value=2, command=grafica, variable=opcion, bg='#A9CCE3',font=('math', 15, 'bold italic')).place(x=110,y=50)
exp = tk.Radiobutton(master=framefun, text='exp(x)', value=3, command=grafica, variable=opcion, bg='#A9CCE3',font=('math', 15, 'bold italic')).place(x=190,y=50)
log = tk.Radiobutton(master=framefun, text='log(x)', value=4, command=grafica, variable=opcion, bg='#A9CCE3',font=('math', 15, 'bold italic')).place(x=70,y=90)
sqrt = tk.Radiobutton(master=framefun, text='sqrt(x)', value=5, command=grafica, variable=opcion, bg='#A9CCE3',font=('math', 15, 'bold italic')).place(x=150,y=90)



'''Colocamos una imagen''' #https://docs.hektorprofe.net/python/interfaces-graficas-con-tkinter/widget-menu/
img=ImageTk.PhotoImage(Image.open("lamborghini-urus.jpg"))
lab=tk.Label(image=img)
lab.place(x=370,y=310)

window.mainloop()




