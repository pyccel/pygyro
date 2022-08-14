from pyccel.decorators import pure


@pure
def ind_to_tp_ind(ir: int, it: int, N_r: int) -> int:
    """
    Convert one-dimensional indices to tensorproduct index

    Parameters
    ----------
        ir : int
            index in r-direction

        it : int
            index in theta-direction

        N_r : int
            number of nodes in r-direction

    Returns
    -------
        tp_ind : int
            tensorproduct index
    """
    tp_ind = ir + it * N_r

    return tp_ind


@pure
def neighbour_index(posr: int, post: int, ir: int, it: int, N_r: int, N_theta: int) -> int:
    """
    Calculate the tensorproduct index given two one-dimensional indices
    and a positional offset with periodic continuation for both variables

    Parameters
    ----------
        posr : int
            position of the neighbour in r-direction relative to index ir

        post : int
            position of the neighbour in theta-direction relative to index it

        ir : int
            index in r-direction of the node of which we want to compute the neighbour index

        it : int
            index in theta-direction of the node of which we want to compute the neighbour index

        N_r : int
            number of nodes in r-direction

        N_theta : int
            number of nodes in theta-direction

    Returns
    -------
        neigh_ind : int
            index of the neighbour
    """
    neigh_ind = (ir + posr) % N_r + ((it + post) % N_theta) * N_r

    return neigh_ind
