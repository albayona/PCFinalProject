#####################################
"""
Lógica - Cálculos
"""
#####################################

##
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
import scipy.optimize as opt

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 150.0

# Average sodoum channel conductance per unit area (mS/cm^2)
gNa = 120.0

# Average potassium channel conductance per unit area (mS/cm^2)
gK = 36.0

# Average leak channel conductance per unit area (mS/cm^2)
gL = 0.3

# Membrane capacitance per unit area (uF/cm^2)
Cm = 1.0

# Sodium potential (mV)
ENa = 50.0

# Potassium potential (mV)
Ek = -77.0

# Leak potential (mV)
EL = -54.4

# Time values
T = np.linspace(tmin, tmax, 10000)

Tbase = 6.3

Q10 = 3.0

# Potassium ion-channel rate functions


# Sodium ion-channel rate functions
def alpha_n(V):
    return (0.01 * (V + 55.0)) / (1.0 - np.exp(-(V + 55.0) / 10.0))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def FV(t, Vm, m, n, h):
    GK = (gK / Cm) * (n**4.0)
    GNa = (gNa / Cm) * (m**3.0) * h
    GL = gL / Cm

    INa = (GNa * (Vm - ENa))
    Ik = (GK * (Vm - Ek))
    Il = (GL * (Vm - EL))

    return (I(t) / Cm) - (INa + Ik + Il)


def FN(Vm, n, T):
    return phi(T) * (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)


def FM(Vm, m, T):
    return phi(T) * (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)


def FH(Vm, h, T):
    return phi(T) * (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)


def phi(T):
    return Q10**((T - Tbase) / 10.0)


def EulerForward(dt, t0, tf, T, V0, m0, n0, h0):
    time = np.arange(t0, tf + dt, dt)
    N = len(time)

    m = np.zeros(N)
    n = np.zeros(N)
    h = np.zeros(N)
    V = np.zeros(N)

    # n, m, and h steady-state values
    '''Initial m - value'''
    m[0] = m0
    '''Initial n - value'''
    n[0] = n0
    ''' Initial h - value  '''
    h[0] = h0

    V[0] = V0

    for t in range(1, N):
        m[t] = m[t - 1] + dt * FM(V[t - 1], m[t - 1], T)
        n[t] = n[t - 1] + dt * FN(V[t - 1], n[t - 1], T)
        h[t] = h[t - 1] + dt * FH(V[t - 1], h[t - 1], T)
        V[t] = V[t -
                 1] + dt * FV(time[t], V[t - 1], m[t - 1], n[t - 1], h[t - 1])

    return time, V


def get_EB(X, yv, ym, yn, yh, Te, dt, t):
    return [
        yv - X[0] + dt * FV(t, X[0], X[1], X[2], X[3]),
        ym - X[1] + dt * FM(X[0], X[1], Te),
        yn - X[2] + dt * FN(X[0], X[2], Te),
        yh - X[3] + dt * FH(X[0], X[3], Te),
    ]


def EulerBackward(dt, t0, tf, T, V0, m0, n0, h0):
    time = np.arange(t0, tf + dt, dt)
    N = len(time)

    m = np.zeros(N)
    n = np.zeros(N)
    h = np.zeros(N)
    V = np.zeros(N)

    # n, m, and h steady-state values
    '''Initial m - value'''
    m[0] = m0
    '''Initial n - value'''
    n[0] = n0
    ''' Initial h - value  '''
    h[0] = h0
    ''' Initial V - value  '''
    V[0] = V0

    for t in range(1, N):
        back_array = opt.fsolve(
            get_EB, np.array([V[t - 1], m[t - 1], n[t - 1], h[t - 1]]),
            (V[t - 1], m[t - 1], n[t - 1], h[t - 1], T, dt, time[t]))

        V[t] = back_array[0]
        m[t] = back_array[1]
        n[t] = back_array[2]
        h[t] = back_array[3]

    return time, V


def RK2(dt, t0, tf, T, V0, m0, n0, h0):
    time = np.arange(t0, tf + dt, dt)
    N = len(time)

    m = np.zeros(N)
    n = np.zeros(N)
    h = np.zeros(N)
    V = np.zeros(N)

    # n, m, and h steady-state values
    '''Initial m - value'''
    m[0] = m0
    '''Initial n - value'''
    n[0] = n0
    ''' Initial h - value  '''
    h[0] = h0

    V[0] = V0

    for i in range(1, N):
        k1M = FM(V[i - 1], m[i - 1], T)
        k2M = FM(V[i - 1] + dt, m[i - 1] + k1M * dt, T)
        m[i] = m[i - 1] + (dt / 2.0) * (k1M + k2M)

        k1N = FN(V[i - 1], n[i - 1], T)
        k2N = FN(V[i - 1] + dt, n[i - 1] + k1N * dt, T)
        n[i] = n[i - 1] + (dt / 2.0) * (k1N + k2N)

        k1H = FH(V[i - 1], h[i - 1], T)
        k2H = FH(V[i - 1] + dt, h[i - 1] + k1H * dt, T)
        h[i] = h[i - 1] + (dt / 2.0) * (k1H + k2H)

        k1V = FV(time[i], V[i - 1], m[i - 1], n[i - 1], h[i - 1])
        k2V = FV(time[i], V[i - 1] + dt, m[i - 1] + k1M * dt,
                 n[i - 1] + k1N * dt, h[i - 1] + k1H * dt)
        V[i] = V[i - 1] + (dt / 2.0) * (k1V + k2V)

    return time, V


def RK4(dt, t0, tf, T, V0, m0, n0, h0):
    time = np.arange(t0, tf + dt, dt)
    N = len(time)

    m = np.zeros(N)
    n = np.zeros(N)
    h = np.zeros(N)
    V = np.zeros(N)

    # n, m, and h steady-state values
    '''Initial m - value'''
    m[0] = m0
    '''Initial n - value'''
    n[0] = n0
    ''' Initial h - value  '''
    h[0] = h0

    V[0] = V0

    for i in range(1, N):
        k1M = FM(V[i - 1], m[i - 1], T)
        k2M = FM(V[i - 1] + 0.5 * dt, m[i - 1] + 0.5 * k1M * dt, T)
        k3M = FM(V[i - 1] + 0.5 * dt, m[i - 1] + 0.5 * k2M * dt, T)
        k4M = FM(V[i - 1] + dt, m[i - 1] + k3M * dt, T)
        m[i] = m[i - 1] + (dt / 6.0) * (k1M + 2.0 * k2M + 2.0 * k3M + k4M)

        k1N = FN(V[i - 1], n[i - 1], T)
        k2N = FN(V[i - 1] + 0.5 * dt, n[i - 1] + 0.5 * k1N * dt, T)
        k3N = FN(V[i - 1] + 0.5 * dt, n[i - 1] + 0.5 * k2N * dt, T)
        k4N = FN(V[i - 1] + dt, n[i - 1] + k3N * dt, T)
        n[i] = n[i - 1] + (dt / 6.0) * (k1N + 2.0 * k2N + 2.0 * k3N + k4N)

        k1H = FH(V[i - 1], h[i - 1], T)
        k2H = FH(V[i - 1] + 0.5 * dt, h[i - 1] + 0.5 * k1H * dt, T)
        k3H = FH(V[i - 1] + 0.5 * dt, h[i - 1] + 0.5 * k2H * dt, T)
        k4H = FH(V[i - 1] + dt, h[i - 1] + k3H * dt, T)
        h[i] = h[i - 1] + (dt / 6.0) * (k1H + 2.0 * k2H + 2.0 * k3H + k4H)

        k1V = FV(time[i], V[i - 1], m[i - 1], n[i - 1], h[i - 1])
        k2V = FV(time[i], V[i - 1] + dt, m[i - 1] + 0.5 * k1M * dt,
                 n[i - 1] + 0.5 * k1N * dt, h[i - 1] + 0.5 * k1H * dt)
        k3V = FV(time[i], V[i - 1] + dt, m[i - 1] + 0.5 * k2M * dt,
                 n[i - 1] + 0.5 * k1N * dt, h[i - 1] + 0.5 * k1H * dt)
        k4V = FV(time[i], V[i - 1] + dt, m[i - 1] + k2M * dt,
                 n[i - 1] + k1N * dt, h[i - 1] + k1H * dt)
        V[i] = V[i - 1] + (dt / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)

    return time, V


# n, m, and h steady-state values


def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))


def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))


def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


# Current
def I(t):
    I = 40.0

    if 0.0 <= t <= 60.0:
        I = 10.0
    elif 70.0 < t < 140.0:
        I = -17.0

    return I


def get_time_voltage_EF():
    time, voltage = EulerForward(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(),
                                 h_inf())
    return time, voltage


def get_time_voltage_EB():
    time, voltage = EulerBackward(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(),
                                  h_inf())
    return time, voltage


def get_time_voltage_RK2():
    time, voltage = RK2(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(), h_inf())
    return time, voltage


def get_time_voltage_RK4():
    time, voltage = RK4(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(), h_inf())
    return time, voltage


#####################################
"""
Interfaz Gráfica
"""
#####################################

from PIL import ImageTk, Image
from matplotlib import style
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

###############
## Window Config
###############

# Colores
sw_night = "#596275"
sw_green = "#05c46b"
sw_lg_green = "#0be881"
sw_lg_white = "#d2dae2"
sw_white = "#ffffff"

# Configuración de la ventana
window = tk.Tk()

appTitle = "Modelo Matemático Hodgkin & Huxley"
window.geometry('1200x700')
window.title(appTitle)

# Frames de la interfaz
input_frame = tk.Frame(master=window)
input_frame.place(x=20, y=20)
input_frame.config(bg=sw_night, width=400, height=660, relief=tk.FLAT)

###############
## Functions
###############

titulo_grafica = {
    "1": "Euler Forward",
    "2": "Euler Backward",
    "3": "Euler Modificado",
    "4": "Runge Kutta 2",
    "5": "Runge Kutta 4",
    "6": "Odeint",
}


def close_app():
    """
    Cierra la aplicación
    """
    window.destroy()


def get_function():
    if opt_tg.get() == 1:
        return get_time_voltage_EF()
    elif opt_tg.get() == 2:
        return get_time_voltage_EB()
    elif opt_tg.get() == 3:
        return get_time_voltage_EB()  # TODO - seria euler mod
    elif opt_tg.get() == 4:
        return get_time_voltage_RK2()
    elif opt_tg.get() == 5:
        return get_time_voltage_RK4()
    else:
        return get_time_voltage_EF()  # TODO - seria odeint


def graficar():
    fig = plt.Figure(figsize=(8, 4), dpi=100)

    time, voltage = get_function()

    fig.add_subplot(111).plot(time, voltage)
    fig.suptitle(titulo_grafica[f"{opt_tg.get()}"])
    plt.close()
    plt.style.use('seaborn-darkgrid')
    Plot = FigureCanvasTkAgg(fig, master=window)
    Plot.draw()
    Plot.get_tk_widget().place(x=420, y=20)


###############
## Buttons
###############

close_btn = ttk.Button(window, text='Salir', command=close_app).place(x=1100,
                                                                      y=660)

###############
## Labels
###############

x_if = 25
y_if = 25
pad_xf = 35
pad_yf = 35

# titulo seleccion tipo gráfica
lbl_params = tk.Label(
    master=input_frame,
    fg=sw_lg_white,
    bg=sw_night,
    text="Tipo de gráfica",
    font=('Arial', 18),
).place(x=x_if, y=y_if)

# selección de tipo de gráfica
opt_tg = tk.IntVar()
eulerFor_tk = tk.Radiobutton(master=input_frame,
                             text="Euler Forward",
                             value=1,
                             variable=opt_tg,
                             bg=sw_night,
                             fg=sw_white,
                             font=('Arial', 18),
                             command=graficar).place(x=x_if, y=y_if + pad_yf)
eulerBack_tk = tk.Radiobutton(master=input_frame,
                              text="Euler Backward",
                              value=2,
                              variable=opt_tg,
                              bg=sw_night,
                              fg=sw_white,
                              font=('Arial', 18),
                              command=graficar).place(x=x_if,
                                                      y=y_if + (pad_yf * 2))
eulerMod_tk = tk.Radiobutton(master=input_frame,
                             text="Euler Modificado",
                             value=3,
                             variable=opt_tg,
                             bg=sw_night,
                             fg=sw_white,
                             font=('Arial', 18),
                             command=graficar).place(x=x_if,
                                                     y=y_if + (pad_yf * 3))
rk2_tk = tk.Radiobutton(master=input_frame,
                        text="Runge Kutta 2",
                        value=4,
                        variable=opt_tg,
                        bg=sw_night,
                        fg=sw_white,
                        font=('Arial', 18),
                        command=graficar).place(x=x_if, y=y_if + (pad_yf * 4))
rk4_tk = tk.Radiobutton(master=input_frame,
                        text="Runge Kutta 4",
                        value=5,
                        variable=opt_tg,
                        bg=sw_night,
                        fg=sw_white,
                        font=('Arial', 18),
                        command=graficar).place(x=x_if, y=y_if + (pad_yf * 5))
odeint_tk = tk.Radiobutton(master=input_frame,
                           text="Odeint",
                           value=6,
                           variable=opt_tg,
                           bg=sw_night,
                           fg=sw_white,
                           font=('Arial', 18),
                           command=graficar).place(x=x_if,
                                                   y=y_if + (pad_yf * 6))

# titulo corriente
lbl_params = tk.Label(
    master=input_frame,
    fg=sw_lg_white,
    bg=sw_night,
    text="Corriente",
    font=('Arial', 18),
).place(x=x_if, y=y_if + (pad_yf * 8))

opt_corriente = tk.IntVar()
corriente_fija_tk = tk.Radiobutton(master=input_frame,
                                   text="Corriente Fija",
                                   value=1,
                                   variable=opt_corriente,
                                   bg=sw_night,
                                   fg=sw_white,
                                   font=('Arial', 18),
                                   command=None).place(x=x_if,
                                                       y=y_if + (pad_yf * 9))
corriente_variable_tk = tk.Radiobutton(master=input_frame,
                                       text="Corriente Variable",
                                       value=1,
                                       variable=opt_corriente,
                                       bg=sw_night,
                                       fg=sw_white,
                                       font=('Arial', 18),
                                       command=None).place(x=x_if,
                                                           y=y_if +
                                                           (pad_yf * 10))

# titulo parámetros
lbl_params = tk.Label(
    master=input_frame,
    fg=sw_lg_white,
    bg=sw_night,
    text="Parámetros",
    font=('Arial', 18),
).place(x=x_if, y=y_if + (pad_yf * 12))

###############
## Window main loop
###############

window.mainloop()
