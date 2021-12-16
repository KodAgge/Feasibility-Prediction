import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('bmh')
mpl.rc('lines', linewidth=2, linestyle='-')
plt.rcParams['figure.dpi'] = 75
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.titlesize'] = 'small'

if __name__=="__main__":
    import numpy as np
    plt.plot(np.random.random(5))
    plt.plot(np.random.random(5),label='plot!')
    plt.scatter(np.random.random(5)*5,np.random.random(5),label='scatter!')
    plt.scatter(np.random.random(5)*5,np.random.random(5))
    plt.scatter(np.random.random(5)*5,np.random.random(5))

    plt.legend(title="title :) ")
    plt.title("Test")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()
