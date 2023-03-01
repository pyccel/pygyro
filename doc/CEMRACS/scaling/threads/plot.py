import numpy as np
import matplotlib.pyplot as plt
import article_setup

p_threads, p_times = np.loadtxt('pygyro.txt').T
s_threads, s_times = np.loadtxt('selalib.txt').T

p_speedup = p_times[0]/p_times
s_speedup = s_times[0]/s_times


p_efficiency = p_speedup/p_threads
s_efficiency = s_speedup/s_threads

maj_ticks = 2**np.arange(0,12,2)
min_ticks = 2**np.arange(0,12)

fig, ax = plt.subplots(1,1)
ax.plot(p_threads, p_times, label='PyGyro')
ax.plot(s_threads, s_times, label='SeLaLib')
ax.plot(maj_ticks, p_times[0]/maj_ticks, '--', label='ideal scaling')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
fig.tight_layout()
fig.savefig('thread_times.png')

fig, ax = plt.subplots(1,1)
ax.plot(p_threads, p_speedup, label='PyGyro')
ax.plot(s_threads, s_speedup, label='SeLaLib')
ax.plot(p_threads, p_threads, '--', label='ideal speedup')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Speedup')
ax.set_xscale('log')
ax.set_yscale('log')
maj_ticks = 2**np.arange(0,12,2)
min_ticks = 2**np.arange(0,12)
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
fig.tight_layout()
fig.savefig('thread_speedup.png')

fig, ax = plt.subplots(1,1)
ax.plot(p_threads, p_efficiency, label='PyGyro')
ax.plot(s_threads, s_efficiency, label='SeLaLib')
ax.set_xlabel('MPI processes')
ax.set_ylabel('Efficiency')
ax.set_xscale('log')
maj_ticks = 2**np.arange(0,12,2)
min_ticks = 2**np.arange(0,12)
ax.set_xticks(maj_ticks, maj_ticks)
ax.set_xticks(min_ticks, [], minor=True)
ax.legend()
ax.set_ylim(0,1.05)
ax.grid()
fig.tight_layout()
fig.savefig('thread_efficiency.png')
plt.show()
