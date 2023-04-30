import numpy as np 
import matplotlib.pyplot as plt
import raphael as cone

#load in all needed data
spins = np.load('spins.npy')

# nms = np.load('num_res_idun/gs_100_100_0.5_0.0_0.0_0.3_2.0.npy')
# sc = np.load('num_res_idun/gs_100_100_0.5_1.0_0.0_0.0_2.0.npy')
ssc = np.load('num_res_idun/gs_80_80_0.5_1.0_0.05_0.2_2.0.npy')
tsc = np.load('num_res_idun/gs_80_80_0.5_0.01_1.0_0.2_2.0.npy')
tsc2 = np.load('num_res_idun/gs_80_80_0.5_0.01_0.8_0.2_2.0.npy')


groundstates = [ssc,tsc, tsc2] #[vnm, sc] #, ssc, tsc]
titles = ['ssc','tsc', 'tsc2']#['vnm', 'csc'] #, 'singlet dominated superconductor', 'triplet dominated superconductor']

#loop over the different systems
for index in range(len(groundstates)):
    data = groundstates[index]
    #extract distances and respective groundstates
    distance = data[:,0]
    gs = data[:,1]

    #determine unique distances and their indicies
    dis_uni = np.unique(distance)
    inter = [np.where(distance==element)[0][0] for element in dis_uni]
    idx_uni = np.array(inter)
    # dis_uni = distance[idx_uni[0]:idx_uni[1]]

    most_deg = np.where(np.diff(idx_uni)==max(np.diff(idx_uni)))[0]

    #take first groundstate for each unique distance
    gs_uni = gs[idx_uni] #gs[idx_uni[most_deg[0]]:idx_uni[most_deg[0]+1]]
    dis_uni = distance[idx_uni]#distance[idx_uni[most_deg[0]]:idx_uni[most_deg[0]+1]]

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
    
    plt.title(titles[index])
    plt.savefig('saga/groundstate/'+titles[index]+'.pdf')
    plt.clf()
    # plt.show()
