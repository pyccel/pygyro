from mpi4py import MPI
import os

def setupSave(constants,foldername: str = None,
                comm: MPI.Comm = MPI.COMM_WORLD, root: int = 0):
    if (comm.Get_rank()==root):
        if (foldername is None):
            i=0
            foldername="simulation_{0}".format(i)
            while(os.path.isdir(foldername)):
                i+=1
                foldername="simulation_{0}".format(i)
            os.mkdir(foldername)
            foldername=comm.bcast(foldername,root=root)
        else:
            if (not os.path.isdir(foldername)):
                os.mkdir(foldername)
        filename = '{0}/initParams.json'.format(foldername)
        print(constants,file=open(filename, "w"))
    else:
        if (foldername is None):
            foldername=comm.bcast("",root=root)
    return foldername
