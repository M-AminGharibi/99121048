import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2*np.pi, 100)

y_sin = np.sin(x)
y_cos = np.cos(x)


plt.plot(x, y_sin, label='Sin')
plt.plot(x, y_cos, label='Cos')


plt.fill_between(x, y_sin, y_cos, where=(y_sin > y_cos), color='gray', alpha=0.3, hatch='//')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
