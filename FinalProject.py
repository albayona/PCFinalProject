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
    GK = (gK / Cm) * (n ** 4.0)
    GNa = (gNa / Cm) * (m ** 3.0) * h
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
    return Q10 ** ((T - Tbase) / 10.0)


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
        V[t] = V[t - 1] + dt * FV(time[t], V[t - 1], m[t - 1], n[t - 1], h[t - 1])

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
            get_EB, np.array([V[t-1], m[t-1], n[t-1], h[t-1]]),
            (V[t - 1], m[t - 1], n[t - 1], h[t - 1], T, dt, time[t]))

        V[t] = back_array[0]
        m[t] = back_array[1]
        n[t] = back_array[2]
        h[t] = back_array[3]

        print(V[t])

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


Time, Voltage = EulerForward(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(),
                             h_inf())

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(Time, Voltage)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Vm (mV)')
ax.set_title('Neuron potential(Euler Forward)')
plt.grid()

plt.show()

Time, Voltage = RK2(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(), h_inf())

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(Time, Voltage)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Vm (mV)')
ax.set_title('Neuron potential (RK2)')
plt.grid()

plt.show()

Time, Voltage = EulerBackward(0.01, tmin, tmax, 6.0, 0.0, m_inf(), n_inf(), h_inf())

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(Time, Voltage)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Vm (mV)')
ax.set_title('Neuron potential (Euler Backward)')
plt.grid()

plt.show()
