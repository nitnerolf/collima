from scipy.special import j1
# plot Airy function J1(x) for x in the range 0 to 20
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
y = (2*j1(x)/x)**2
y[0]=1 # Handle the singularity at x=0

plt.figure(1)
plt.plot(x, y, label=f'Airy function')
plt.title("Airy Function J1(x)")

lim_bij = 0.0045*4
plt.axhline(y=lim_bij, color='r', linestyle='--', label='Limit bij')

nw = np.where(y > lim_bij)[0]
lim_bij = x[nw][-1]
print('bijective limit', lim_bij)

plt.figure(2)
plt.plot(y[x <= lim_bij],x[x <= lim_bij], label='Airy function inverse')
plt.title("Inverse Airy Function J1(x)")

# Fit a polynomial of specified order within the bijective limit
order = 10  # Set the desired polynomial order here

# Select data within the bijective limit
x_fit = x[x <= lim_bij]
y_fit = y[x <= lim_bij]

x_fit2 = x_fit - np.mean(x_fit)  # Center the y_fit data around zero
y_fit2 = y_fit - np.mean(y_fit)  # Center the y_fit data around zero

print('x_fit', x_fit)
print('y_fit', y_fit)

# Fit polynomial
coeffs = np.polyfit(x_fit, y_fit, order)
coeffs2 = np.polyfit(y_fit, x_fit, order)
print('coeffs', coeffs)
print('coeffs2', coeffs2)
poly  = np.poly1d(coeffs)
poly2 = np.poly1d(coeffs2)

plt.figure(1)
# Plot the polynomial fit
plt.plot(x, poly(x), label=f'Poly fit (order {order})', linestyle='--',color='red')
plt.ylim(0, 1)

print('polyn',poly(x_fit))
plt.legend()


plt.figure(2)
# Plot the polynomial fit
plt.plot(y, poly2(y), label=f'Poly fit (order {order})', linestyle='--',color='red')

plt.figure(3)
# Plot the polynomial fit
plt.plot(y[x <= lim_bij], x[x <= lim_bij]-poly2(y[x <= lim_bij]), label=f'Poly fit (order {order})', linestyle='--',color='red')


def get_inverse_airy(x):
    """
    Returns the inverse of the Airy function J1(x) for x in the range [0, lim_bij].
    """
    coeffs = [
        -2.82669678e+03,  1.27516534e+04, -2.39643499e+04,  2.40786554e+04,
        -1.36276857e+04,  3.97022426e+03, -2.31741625e+02, -2.10699163e+02,
        6.97641625e+01, -1.25551142e+01,  3.49086140e+00
    ]
    pol = np.poly1d(coeffs)
    return pol(x)


plt.figure(2)
plt.plot(y[x <= lim_bij], get_inverse_airy(y[x <= lim_bij]), label=f'abaque function', linestyle='--',color='green')


plt.show()