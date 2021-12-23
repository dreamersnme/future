import numpy as np
import time
import matplotlib.pyplot as plt



x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.ion()


figure = plt.figure(figsize=(5, 8))
ax = figure.add_subplot(211)
line1, = ax.plot([], [],'g', label='Account cash balance', alpha=0.8)

plt.title("Dynamic Plot of sinx", fontsize=25)

plt.xlabel("X", fontsize=18)
plt.ylabel("sinX", fontsize=18)
plt.legend()


for p in range(100):
    updated_y = np.cos(x - 0.05 * p)

    line1.set_xdata(x)
    line1.set_ydata(updated_y)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(updated_y), max(updated_y))

    figure.canvas.draw()

    figure.canvas.flush_events()
    time.sleep(0.1)