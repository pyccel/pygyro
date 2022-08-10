from .process_grid import compute_2d_process_grid, compute_2d_process_grid_from_max
import pytest


@pytest.mark.serial
def test_compute_2d_process_grid():
    """
    TODO
    """
    with pytest.raises(RuntimeError):
        print(compute_2d_process_grid_from_max(2, 3, 5))

    assert (compute_2d_process_grid_from_max(1, 6, 6) == (1, 6))
    assert (compute_2d_process_grid_from_max(6, 6, 6) == (2, 3))
    assert (compute_2d_process_grid_from_max(2, 3, 6) == (2, 3))
    assert (compute_2d_process_grid_from_max(3, 2, 6) == (3, 2))
    assert (compute_2d_process_grid_from_max(6, 1, 6) == (6, 1))
    assert (compute_2d_process_grid_from_max(1200, 500, 60) == (12, 5))
    assert (compute_2d_process_grid_from_max(500, 1200, 60) == (5, 12))
    assert (compute_2d_process_grid_from_max(1200, 1200, 60) == (6, 10))

    assert (compute_2d_process_grid([256, 512, 32, 128], 4) == (4, 1))
    assert (compute_2d_process_grid([256, 512, 32, 128], 8) == (4, 2))
    assert (compute_2d_process_grid([256, 512, 32, 128], 16) == (8, 2))
    assert (compute_2d_process_grid([256, 512, 32, 128], 32) == (8, 4))
    assert (compute_2d_process_grid([256, 512, 32, 128], 64) == (16, 4))
    assert (compute_2d_process_grid([256, 512, 32, 128], 9) == (9, 1))
    assert (compute_2d_process_grid([256, 512, 32, 128], 18) == (9, 2))
    assert (compute_2d_process_grid([256, 512, 32, 128], 100) == (20, 5))
