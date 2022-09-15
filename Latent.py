from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from newVaes import _get_dataset
import numpy as np
from scipy.stats import norm

cdict = {'red':   ((0.0,  0.0, 0.8),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.95),
                   (1.0,  0.25, 1.0)),

         'blue':  ((0.0,  0.0, 1.0),
                   (1.0,  0.5, 1.0))}
DarkBlue = LinearSegmentedColormap('DarkBlue', cdict)

def plotLatent1d(vae, q, latent_dim, seqs, paths, name):
    z_mean, z_log_var, z = vae.encoder.predict(_get_dataset(seqs, 10000, q, training=False))
    st = np.exp(z_log_var/2)

    # make 1d distribution plots
    fig = plt.figure(figsize=(12,12))
    nx = max(latent_dim//2, 1)
    ny = (latent_dim-1)//nx + 1

    for z1 in range(latent_dim):
        fig.add_subplot(nx,ny,z1+1)
        h, b, _ = plt.hist(z_mean[:,z1], bins=100, density=True)
        wm, ws = z_mean[0][z1], st[0][z1]
        x = np.linspace(wm - 5*ws, wm + 5*ws, 200)
        y = norm.pdf(x, wm, ws)
        y = y*np.max(h)/np.max(y)
        plt.plot(x, y, 'r-')
        plt.xlim(-5,5)
        plt.title('Z{}, <z{}>_std={:.2f}'.format(z1, z1, np.std(z_mean[:,z1])))

    plt.savefig(paths["vis_path"] + "LatentTraining_1d_{}.png".format(name))
    plt.close()

    return z_mean, st

def plotLatent2d(z_mean, st, latent_dim, paths, name):

    r = 4
    s = np.linspace(-r, r, 50)
    X, Y = np.meshgrid(s, s)
    red = np.broadcast_to(np.array([1.,0, 0, 1]), (len(s), len(s), 4)).copy()

    fig = plt.figure(figsize=(12,12))
    counter = 0

    for z1 in range(latent_dim):
        print('Var z{}: {}'.format(z1, np.var(z_mean[:, z1])))
        for z2 in range(z1+1,latent_dim):
            counter += 1
            fig.add_subplot(latent_dim-1,latent_dim-1,counter)

            plt.hist2d(z_mean[:, z2], z_mean[:, z1],
                    bins=np.linspace(-r,r,50), cmap=DarkBlue, cmin=1)

            nn = (norm.pdf(X, z_mean[0][z2], st[0][z2]) *
                norm.pdf(Y, z_mean[0][z1], st[0][z1]))
            nn = nn/np.max(nn)/1.5
            red[:,:,3] = nn
            plt.imshow(red, extent=(-r,r,-r,r), origin='lower', zorder=2)

            ##wildtype in red
            #plt.scatter(m[0][z1], m[0][z2],c="r", alpha=1)
            # make 1std oval for wt
            #wtv = Ellipse((m[0][z2],  m[0][z1]),
            #              width=st[0][z2], height=st[0][z1],
            #              facecolor='none', edgecolor='red', lw=2)
            #plt.gca().add_patch(wtv)
            #wtv = Ellipse((m[0][z2],  m[0][z1]),
            #              width=2*st[0][z2], height=2*st[0][z1],
            #              facecolor='none', edgecolor='red', lw=1)
            #plt.gca().add_patch(wtv)
            plt.xlim(-4,4)
            plt.ylim(-4,4)

            fs = 26
            if latent_dim <= 7:
                fs *= 2
            if z1 == 0:
                plt.xlabel('$z_{{{}}}$'.format(z2), fontsize=fs, labelpad=fs/2)
                plt.gca().xaxis.set_label_position('top')
            if z2 == latent_dim -1:
                plt.ylabel('$z_{{{}}}$'.format(z1), fontsize=fs)
                plt.gca().yaxis.set_label_position('right')

            plt.xticks([])
            plt.yticks([])
        counter += z1+1

    plt.subplots_adjust(right=0.92, bottom=0.01, left=0.01, top=0.92)
    plt.savefig(paths["vis_path"] + "LatentTraining_2d_{}.png".format(name))
    plt.close()