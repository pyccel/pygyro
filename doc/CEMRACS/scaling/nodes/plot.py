import numpy as np
import matplotlib.pyplot as plt
import article_setup

p_nodes, p_times = np.loadtxt('pygyro.txt').T
s_nodes, s_times = np.loadtxt('selalib.txt').T

p_speedup = p_times[0]/p_times
s_speedup = s_times[0]/s_times

p_efficiency = p_speedup/p_nodes
s_efficiency = s_speedup/s_nodes

p_nodes *= 32
s_nodes *= 32

maj_ticks = 32*2**np.arange(0,8,2)
min_ticks = 32*2**np.arange(0,8)

fig, ax = plt.subplots(1,1)
ax.plot(p_nodes, p_times, label='PyGyro')
ax.plot(s_nodes, s_times, label='SeLaLib')
ax.plot(maj_ticks, p_times[0]/(2**np.arange(0,8,2)), '--', label='ideal scaling')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
fig.tight_layout()
fig.savefig('node_times.png')

fig, ax = plt.subplots(1,1)
ax.plot(p_nodes, p_speedup, label='PyGyro')
ax.plot(s_nodes, s_speedup, label='SeLaLib')
ax.plot(p_nodes, p_nodes/32, '--', label='ideal speedup')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Speedup')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
fig.tight_layout()
fig.savefig('node_speedup.png')

fig, ax = plt.subplots(1,1)
ax.plot(p_nodes, p_efficiency, label='PyGyro')
ax.plot(s_nodes, s_efficiency, label='SeLaLib')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Efficiency')
ax.set_xscale('log')
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
ax.set_ylim(0,1.05)
ax.grid()
fig.tight_layout()
fig.savefig('node_efficiency.png')
plt.show()
