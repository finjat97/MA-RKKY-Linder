import numpy as np
import matplotlib.pyplot as plt

details= '_100_-5e-01_0.15_0.25_0.25'
## load in data
J, I, D, G = [], [], [], []
with open('ana_res_final/ana_J'+details+'.txt', 'r') as file:
    for row in file:
        for element in row.split():
            J += [complex(element)]
inter = []
with open('ana_res_final/ana_I'+details+'.txt', 'r') as file:
    for row in file:
        for element in row.split():
            inter += [complex(element)]
        I += [inter]
        inter = []

inter = []
counter = 0
with open('ana_res_final/ana_D'+details+'.txt', 'r') as file:
    for row in file:
        counter += 1
        if counter != 3:
            for element in row.split():
                inter += [complex(element)]
            D += [inter]
            inter = []
        else:
            inter = np.zeros(151, dtype='complex128')
            D += [inter]

inter = []
counter = 0
with open('ana_res_final/ana_G'+details+'.txt', 'r') as file:
    for row in file:
        for element in row.split():
            G += [complex(element)]
        # G += [inter]
        # inter= []
    
J = np.asarray(J)
I = np.asarray(I)
D = np.asarray(D)
G = np.asarray(G)

# print(J.shape, I.shape, D.shape, G.shape)

x = np.arange(0,15.1,0.1)

# max = []
# for idx in range(x.shape[0]):
#     inter = np.max([abs(J[idx]/10), abs(I[0][idx]), abs(I[1][idx]), abs(D[1][idx]), abs(G[idx])])
#     max += [inter]
# plt.scatter(x,max, marker='x')

plt.plot(x,J.real, label='J')

plt.plot(x,G.real,':', label='xy')

plt.plot(x,I[0].real, label='Ix')
plt.plot(x,I[1].real, label='Iy')
plt.plot(x,I[2].real, label='Iz')
plt.plot(x,D[0].real, label='Dx')
plt.plot(x,D[1].real, label='Dy')
plt.plot(x,D[2].real, label='Dz')

plt.legend(fontsize=10)
para = details.split('_')[1:]
plt.title('Analytical coefficients for \n'+r' $\Gamma$ for N= '+str(para[0])+r' $\mu$= '+str(para[1])+r' $\gamma$ = '+str(para[2])+r' $\Delta_s$ = '+str(para[3])+r' $\Delta_t$ = '+str(para[4]), fontsize=14)
plt.xlabel('distance in a', fontsize=14)
plt.ylabel('energy in 1/t', fontsize=14)
plt.ylim((-6000, 5000))
plt.grid()
plt.tight_layout()
plt.savefig('ana_coeffi'+details+'.pdf')
plt.show()