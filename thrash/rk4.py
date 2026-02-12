import numpy as np
import matplotlib.pyplot as plt

def model(y, t, omega_sq):
    # y to wektor stanu: [y1, y2] gdzie y1 = pozycja, y2 = prędkość
    y1, y2 = y

    dy1_dt = y2
    dy2_dt = -omega_sq * y1

    return np.array([dy1_dt, dy2_dt])

def rk4_step(f, y, t, h, omega_sq):
    """
    Wykonuje jeden krok metody Rungego-Kutty 4 rzędu.
    f - funkcja modelu
    y - obecny stan
    t - obecny czas
    h - rozmiar kroku
    """
    k1 = f(y, t, omega_sq)
    k2 = f(y + h * k1 / 2.0, t + h / 2.0, omega_sq)
    k3 = f(y + h * k2 / 2.0, t + h / 2.0, omega_sq)
    k4 = f(y + h * k3, t + h, omega_sq)

    # Uśredniony krok
    y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next

# Parametry symulacji
omega = 0.5
h = 0.1
t_max = 20.0

# Warunki początkowe
y = np.array([1.0, 0.0])
t = 0.0

t_values = []
y_values = []

# Symulacja
while t <= t_max:
    y = rk4_step(model, y, t, h, omega)
    t += h

    t_values.append(t)
    y_values.append(y)

# Konwersja wyników na tablice numpy
results = np.array(y_values)
position = results[:, 0]
velocity = results[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(t_values, position, label='Pozycja (y1)', marker='o', markersize=4)
plt.plot(t_values, velocity, label="Prędkość (v)", linestyle='--')
plt.title('Ruch harmoniczny prosty - Metoda Rungego-Kutty 4 rzędu')
plt.xlabel('Czas')
plt.ylabel('Wartość')
plt.legend()
plt.grid(True)
plt.show()