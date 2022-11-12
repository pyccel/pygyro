from mpi4py import MPI
import numpy as np

from ..model.grid import Grid
from .norms import l2, l1, nParticles
from .energy import KineticEnergy, PotentialEnergy


class DiagnosticCollector:
    """
    TODO
    """

    def __init__(self, comm, saveStep: int, dt: float, distribFunc: Grid, phi: Grid, constants):
        self.saveStep = saveStep
        self.dt = dt
        self.comm = comm
        self.rank = comm.Get_rank()

        self.diagnostics = np.zeros([8, saveStep])
        self.l2PhiResult = np.zeros(saveStep)
        self.l2GridResult = np.zeros(saveStep)
        self.l1Result = np.zeros(saveStep)
        self.nPartResult = np.zeros(saveStep)
        self.min_val = np.zeros(saveStep)
        self.max_val = np.zeros(saveStep)
        self.KE_val = np.zeros(saveStep)

        self.l2_phi_class = l2(phi.eta_grid, phi.getLayout('v_parallel_2d'))
        self.l1class = l1(distribFunc.eta_grid,
                          distribFunc.getLayout('v_parallel'))
        self.l2_grid_class = l2(distribFunc.eta_grid,
                                distribFunc.getLayout('v_parallel'))
        self.npart = nParticles(distribFunc.eta_grid,
                                distribFunc.getLayout('v_parallel'))
        self.KEclass = KineticEnergy(
            distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)

    def collect(self, f: Grid, phi: Grid, t: float):
        """
        Collect various diagnostics to be saved in the following order:
         0 - time
         1 - l2 norm of the electric potential
         2 - l2 norm of the distribution function
         3 - l1 norm of the distribution function
         4 - number of particles
         5 - minimum value of the distribution function
         6 - maximum value of the distribution function
         7 - Kinetic Energy

        Parameters
        ----------
            f : Grid
                The distribution function

            phi : Grid
                The electric potential

            t : float
                The current time
        """
        ti = t//self.dt
        idx = ti % self.saveStep

        self.diagnostics[0, idx] = t
        self.diagnostics[1, idx] = self.l2_phi_class.l2NormSquared(phi)
        self.diagnostics[2, idx] = self.l2_grid_class.l2NormSquared(f)
        self.diagnostics[3, idx] = self.l1class.l1Norm(f)
        self.diagnostics[4, idx] = self.npart.getN(f)
        self.diagnostics[5, idx] = f.getMin()
        self.diagnostics[6, idx] = f.getMax()
        self.diagnostics[7, idx] = self.KEclass.getKE(f)

    def reduce(self):
        """
        TODO
        """
        self.comm.Reduce(self.diagnostics[1, :],
                         self.l2PhiResult, op=MPI.SUM, root=0)
        self.comm.Reduce(self.diagnostics[2, :],
                         self.l2GridResult, op=MPI.SUM, root=0)
        self.comm.Reduce(self.diagnostics[3, :],
                         self.l1Result, op=MPI.SUM, root=0)
        self.comm.Reduce(self.diagnostics[4, :],
                         self.nPartResult, op=MPI.SUM, root=0)
        self.comm.Reduce(self.diagnostics[5, :],
                         self.min_val, op=MPI.MIN, root=0)
        self.comm.Reduce(self.diagnostics[6, :],
                         self.max_val, op=MPI.MAX, root=0)
        self.comm.Reduce(self.diagnostics[7, :],
                         self.KE_val, op=MPI.SUM, root=0)
        if (self.rank == 0):
            self.l2PhiResult = np.sqrt(self.l2PhiResult)
            self.l2GridResult = np.sqrt(self.l2GridResult)

    def __str__(self):
        """
        TODO
        """
        myStr = []
        for i in range(self.saveStep):
            myStr.append(self.getLine(i))
            myStr.append('\n')
        return ''.join(myStr)

    def getLine(self, i):
        """
        TODO
        """
        return "{t:10g}   {l2P:16.10e}   {l2G:16.10e}   {l1:16.10e}   {np:16.10e}   {minim:16.10e}   {maxim:16.10e}   {ke:16.10e}". \
            format(t=self.diagnostics[0, i], l2P=self.l2PhiResult[i], l2G=self.l2GridResult[i],
                   l1=self.l1Result[i], np=self.nPartResult[i],
                   minim=self.min_val[i], maxim=self.max_val[i], ke=self.KE_val[i])


class AdvectionDiagnostics:
    """
    A class to collect the diagnostics for the advection step. Mass, l2-norm, and
    energy are collected before and after the advection step to compare methods.
    The diagnostics attribute contains:
        0 : mass
        1 : l2-norm
        2 : potential energy
        3 : kinetic energy
    """

    def __init__(self, comm, dt: float, distribFunc: Grid, constants):
        self.dt = dt
        self.comm = comm
        self.rank = comm.Get_rank()

        self.diagnostics = np.zeros(2, dtype=float)
        self.diagnostics_val = np.array([[0.]] * 2)

        self.KEclass = KineticEnergy(
            distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)
        self.PEclass = PotentialEnergy(
            distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)
        # self.PEclass = PotentialEnergy(distribFunc.eta_grid)

    def collect_kinetic_energy(self, f):
        """
        TODO
        """
        self.diagnostics[1] = self.KEclass.getKE(f)

    def collect_potential_energy(self, f, phi):
        """
        TODO
        """
        self.diagnostics[0] = self.PEclass.getPE(f, phi)

    def reduce(self):
        """
        TODO
        """
        for k in range(len(self.diagnostics)):
            self.comm.Reduce(self.diagnostics[k],
                            self.diagnostics_val[k], op=MPI.SUM, root=0)
            # if self.rank == 0:
            #     print(f'k = {k} : {self.diagnostics_val[k][0]}')
