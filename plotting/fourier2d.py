import numpy as np

# -------------------------------------------------
# Personal FFT2D function
# -------------------------------------------------


def Fourier2D(F0, y0, x0):
    """ Personal FFT2D function"""
    nx0 = len(x0)
    ny0 = len(y0)

    assert (ny0, nx0) == F0.shape

    nx = 2 * int(nx0 / 2)
    hnx = int(nx / 2)
    ny = 2 * int(ny0 / 2)
    hny = int(ny / 2)

    x = x0[0:nx]
    y = y0[0:ny]
    F = F0[0:ny, 0:nx]

    Lx = x[nx - 1] - x[0]
    dx = x[1] - x[0]
    kx = np.zeros(nx)
    temp = -np.r_[1:hnx + 1]
    kx[0:hnx] = temp[::-1]
    kx[hnx:nx] = np.r_[0:hnx]

    Ly = y[ny - 1] - y[0]
    dy = y[1] - y[0]
    ky = np.zeros(ny)
    temp = -np.r_[1:hny + 1]
    ky[0:hny] = temp[::-1]
    ky[hny:ny] = np.r_[0:hny]

    TFF = np.zeros((ny, nx), dtype=complex)
    AA = np.zeros((ny, nx), dtype=complex)
    var = np.conjugate(np.fft.fft2(np.conjugate(F))) / float((nx * ny))

    AA[:, 0:hnx] = var[:, hnx:nx]
    AA[:, hnx:nx] = var[:, 0:hnx]
    TFF[0:hny, :] = AA[hny:ny, :]
    TFF[hny:ny, :] = AA[0:hny, :]

    return TFF, kx, ky
