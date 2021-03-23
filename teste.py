import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b):
    return a*(x**b)


massa_zero = 5
errmassa_zero = 0.05 # erro instrumental da balança prof

values1 = np.array([4.6, 4, 4.4, 4, 4.7, 3.9, 4.3, 4.3])
values2 = np.array([2.6, 2.5, 2.7, 3.1, 2.5, 2.8, 2.9, 3])
values3 = np.array([1.9, 2.4, 2.3, 1.9, 2.3, 1.8, 2, 1.7])
values4 = np.array([1.6, 1.4, 1.5, 1.3, 1.4, 1.5, 1.5, 1.3])
values5 = np.array([1.3, 1.1, 1, 0.8, 1.2, 0.9, 0.9, 1])
values6 = np.array([0.6, 0.7, 0.7, 0.6, 0.7, 0.8, 0.6, 0.6])
values7 = np.array([0.5, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.6])
values8 = np.array([0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4])
values9 = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2])
values10 = np.array([0, 0, 0, 0.1, 0.2, 0.1, 0, 0.1])

dadosdiam = [values1,values2,values3,values4,values5,values6,values7,values8,values9,values10]

diam_medio = [np.average(diam) for diam in dadosdiam]
massa = [massa_zero/(2**i) for i in range(10)]
errdiam = [np.std(diam/np.sqrt(len(diam))) for diam in dadosdiam]
errmassa = [errmassa_zero/(2**(i)) for i in range(10)]

initial_guess = [2.5, 2.5]
ans, err = curve_fit(func, diam_medio, massa, initial_guess)

plt.title('Fractais bolinha de papel')
plt.ylabel('Massa (g)')
plt.xlabel('Diâmetro médio (cm)')


plt.plot(diam_medio, massa, marker='o', ls='', color='black', label='Dados experimentais')

plt.errorbar(diam_medio, massa, xerr=errdiam,  yerr=errmassa, fmt=' ',
             color='black')

plt.plot(np.arange(0.0, 5.0, 0.01), func(np.arange(0.0, 5.0, 0.01), *ans), 'r',
         label= f"Parâmetros de ajuste função a*(x**b): \n a = {ans[0]:.4f} +/- {err[0,0]:.4f}"
         f"\n b = {ans[1]:.4f} +/- {err[1,1]:.4f}")

quadrado_da_soma = sum(((massa-func(diam_medio, *ans)/errmassa))**2)
plt.figtext(0.135, 0.73, f'chi-square: {quadrado_da_soma:.2f}')

plt.legend()
plt.show()
