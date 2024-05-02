import matplotlib.pyplot as plt

values = [0.1777,0.1686,0.1636,0.1615,0.1565,0.1529,0.15220, 0.1540,0.1493,0.14666]

plt.plot(values)
base = 0.1331
# plot a line in the same plt with the value is the same as base
plt.plot([0,10],[base,base])
plt.savefig('pics/plot.png')
