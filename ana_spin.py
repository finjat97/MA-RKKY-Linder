import numpy as np 
import matplotlib.pyplot as plt

spins = np.load('spins.npy')

details= '_100_-5e-01_0.05_0.0_0.0'
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

heisenberg = spins[:,0]*spins[:,1]
dm = np.cross(spins[:,0], spins[:,1])
xy = spins[:,0,0]*spins[:,0,1]
yx = spins[:,0,1]*spins[:,0,0]

h = np.array([np.sum(element*heisenberg, axis=1) for element in J])
i = np.array([np.dot(element,heisenberg.T) for element in I.T])
d = np.array([np.dot(element,dm.T) for element in D.T])
g = np.array([element*xy + element*yx for element in G])

free_energy = h+i+d+g

gs = np.array([np.where(element == min(element)) for element in free_energy])
gs_uni = [element[0][0] for element in gs]
dis_uni = np.arange(0,15.1,0.1)

config = spins[gs_uni]

ax = plt.figure().add_subplot(projection='3d')

x = np.asarray(dis_uni) #np.arange(len(dis_uni))#
y = np.zeros(len(gs_uni))
z = y

u = config[:, 0, 0]
v = config[:, 0, 1]
w = config[:, 0, 2]

u2 = config[:, 1, 0]
v2 = config[:, 1, 1]
w2 = config[:, 1, 2]

# cone.plot(x,y,z,u2,v2,w2)

ax.quiver(x, y, z, u, v, w, length=0.6, normalize=True, label='hallo')
ax.quiver(x, y, z, u2, v2, w2, length=0.6, normalize=True, color='green')

# ax.plot(distance*2, y, z, color='black', linewidth=0.2)

#plot starting points of arrows
ax.scatter(
    xs=x,
    ys=y,
    zs=z,
    marker='o',
)

ax.set(ylim=(-0.7,0.7), zlim=(-0.7,0.7))
ax.set_xlabel('distance in a')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.title(titles[index])
plt.savefig('ana_gs'+details+'.pdf')
plt.show()

# dummy = True