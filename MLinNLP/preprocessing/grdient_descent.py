import numpy as np
import matplotlib.pyplot as plt


def f(x):
	y = x**2 - 3.5*x + 1.5
	return y

x = np.arange(-1, 5, 0.01)
y = f(x)

fig, ax  = plt.subplots()
ax.plot(x, y, lw = 0.9, color = 'k')
ax.set_xlim([min(x), max(x)])
ax.set_ylim([-3, max(y)+1])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x_init = -0.8
lr = 0.15
epoch = 10

def gradient_descent(prev_x, lr, epoch):
	x_gd = []
	y_gd = []
	x_gd.append(prev_x)
	y_gd.append(f(prev_x))

	for i in range(epoch):
		current_x = prev_x - lr*(2*prev_x - 3.5)
		x_gd.append(current_x)
		y_gd.append(f(current_x))

		prev_x = current_x

	return x_gd, y_gd

x_gd, y_gd = gradient_descent(x_init, lr, epoch)

fig, ax  = plt.subplots()
ax.plot(x, y, lw = 0.9, color = 'k')
ax.set_xlim([min(x), max(x)])
ax.set_ylim([-3, max(y)+1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.scatter(x_gd, y_gd, c = 'b')
for i in range(1, epoch+1):
	ax.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
	arrowprops=dict(arrowstyle =  '->', color='r',lw=1),
	va = 'center', ha = 'center')
plt.show()


