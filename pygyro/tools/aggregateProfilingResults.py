import argparse

parser = argparse.ArgumentParser(description='Process foldername')
parser.add_argument('foldername', metavar='foldername', nargs=1, type=str,
                    help='folder containing profiling results')
parser.add_argument('nprocs', metavar='nprocs', nargs=1, type=int,
                    help='number of processes used')

args = parser.parse_args()
foldername = args.foldername[0]
nprocs = args.nprocs[0]

# ~ foldername = 'profile/pyc_7_YamanAcc/'
# ~ nprocs=3

stem = 'l2Test'

filename = foldername + stem + '0.txt'

lines = []
file = open(filename, 'r')

start_lines = []

for i in range(5):
    start_lines.append(file.readline())

for line in file:
    results = line.split(maxsplit=5)
    if (len(results) > 0):
        div = results[0].split('/')
        results[0] = div[0]
        lines.append(
            (results[5], [float(results[1]), float(results[3]), int(results[0])]))

file.close()

content = dict(lines)

for i in range(1, nprocs):
    filename = foldername + stem + '{}.txt'.format(i)
    file = open(filename, 'r')
    for j in range(5):
        file.readline()

    for line in file:
        results = line.split(maxsplit=5)
        if (len(results) > 0):
            div = results[0].split('/')
            results[0] = div[0]
            content[results[5]][0] += float(results[1])
            content[results[5]][1] += float(results[3])
            content[results[5]][2] += int(results[0])
    file.close()

results = []
for func, res in content.items():
    results.append([res[2]//nprocs, res[0]/nprocs, res[0] /
                   res[2], res[1]/nprocs, res[1]/res[2], func])

results.sort(key=lambda x: -x[1])

filename = foldername + stem + 'Aggregate.txt'
file = open(filename, 'w')
for line in start_lines[:4]:
    file.write(line)
file.write(
    '   ncalls    tottime    percall    cumtime    percall filename:lineno(function)\n')
for r in results:
    file.write("{:9d}  {:9.3f}  {:9.3f}  {:9.3f}  {:9.3f} {}".format(*r))
file.close()


for r in results[:7]:
    print("{:.3f} & {} & {:.3f} & {:.3f}".format(r[1], r[0], r[2], r[3]))
