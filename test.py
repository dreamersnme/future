import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.ion()
fig = plt.figure(figsize=(8, 9))
# line, = ax.boxplot(np.arange(10))  <-- not needed it seems
ax3 = plt.subplot(211)
ax3.plot([2,4,3,4,3,4])

ax = plt.subplot(212)

ax.boxplot([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,15], [1,4,5], [6,7]], flierprops={'marker': '+', 'markerfacecolor':'r'})

fig.canvas.draw()
fig.canvas.flush_events()

time.sleep(3)
ax.cla()
ax.boxplot([[6,7], [1,4,5], [1,5]])
fig.canvas.draw()
fig.canvas.flush_events()
time.sleep(3)