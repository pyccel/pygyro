
class Result:
    """
    TODO
    """

    def __init__(self, line):
        vals = line.split()
        self._nprocs = int(vals[0])
        self._loop = float(vals[1])
        self._output = float(vals[2])
        self._setup = float(vals[3])
        self._additional = float(vals[4])

        self._nnodes = self._nprocs/32

    def __lt__(self, other):
        """
        TODO
        """
        assert (type(other).__name__ == 'Result')
        return self._nprocs < other._nprocs

    def __eq__(self, n):
        """
        TODO
        """
        return self._nprocs == n

    def __str__(self):
        """
        TODO
        """
        return "{nprocs:8}   {loop:16.10e}   {output:16.10e}   {setup:16.10e}   {additional:16.10e}". \
            format(nprocs=self._nprocs, loop=self._loop, output=self._output,
                   setup=self._setup, additional=self._additional)


results = []

file_object = open("results.txt", "r")
for line in file_object:
    results.append(Result(line))
file_object.close()

results.sort()

if (32 in results):
    R0 = results[results.index(32)]
else:
    R0 = results[0]

file_object = open("scaling.txt", "w")
file_object.write("{:8}   {:16.10}   {:16.10}   {:16.10}   {:16.10}   {:16.10}   {:16.10}\n".
                  format("nprocs", "loop", "output", "setup", "additional", "speedup", "efficiency"))
for r in results:
    file_object.write(r.__str__())
    s = R0._loop/r._loop
    e = s*R0._nnodes/r._nnodes
    file_object.write("   {speedup:16.10e}   {efficiency:16.10e}\n".
                      format(speedup=s, efficiency=e))

file_object.close()
