import numpy as np
import scipy.sparse as sparse

def neighbor_index(pos0, pos1, i0, i1, N0_nodes):
    return (i0+pos0)%N0_nodes + ((i1+pos1)%N0_nodes)*N0_nodes    


def assemble_Jpp(phi_hh, N0_nodes): 
    """
    assemble J_++(phi, . ) as sparse matrix 
    
    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes*N0_nodes

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii%N0_nodes
        i1 = ii//N0_nodes
        
        # .. terms (phi_0+ - phi_0-)/4h^2 * (f_+0 - f_-0)
        coef  = phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes)]
        coef -= phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms -(phi_+0 - phi_-0)/4h^2 * (f_0+ - f_0-)
        coef = -phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes)]
        coef += phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes))
        data.append(coef)
        
        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

def assemble_Jpx(phi_hh, N0_nodes): 
    """
    assemble J_+x(phi, . ) as sparse matrix 
    
    phi_hh: phi/(2h)**2
    """
    
    N_nodes = N0_nodes*N0_nodes

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii%N0_nodes
        i1 = ii//N0_nodes
        
        # .. terms phi_++/4h^2 * (f_0+ - f_+0)
        coef = phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms -phi_--/4h^2 * (f_-0 - f_0-)
        coef = -phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms -phi_-+/4h^2 * (f_0+ - f_-0)
        coef = -phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms phi_+-/4h^2 * (f_+0 - f_0-)
        coef = phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp(phi_hh, N0_nodes): 
    """
    assemble J_x+(phi, . ) as sparse matrix 
    
    phi_hh: phi/(2h)**2
    """
    
    N_nodes = N0_nodes*N0_nodes

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii%N0_nodes
        i1 = ii//N0_nodes
        
        # .. terms phi_+0/4h^2 * (f_++ - f_+-)
        coef = phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms -phi_-0/4h^2 * (f_-+ - f_--)
        coef = -phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms -phi_0+/4h^2 * (f_++ - f_-+)
        coef = -phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes))
        data.append(-coef)

        # .. terms phi_0-/4h^2 * (f_+- - f_--)
        coef = phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes)]
        
        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
