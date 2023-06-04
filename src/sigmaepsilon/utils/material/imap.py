import numpy as np
from numpy import ndarray


def _index_map_4d_to_2d(imap_2d_to_1d: dict) -> dict:
    indices = np.indices((6, 6))
    it = np.nditer([*indices], ["multi_index"])
    imap2d = dict()
    for _ in it:
        i, j = it.multi_index
        imap2d[(i, j)] = imap_2d_to_1d[i] + imap_2d_to_1d[j]
    return imap2d


_imap_2d_to_1d = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}
_imap_4d_to_2d = _index_map_4d_to_2d(_imap_2d_to_1d)


def _map_3x3_to_6x1(arr: ndarray, imap_2d_to_1d: dict = _imap_2d_to_1d) -> ndarray:
    shape_in = arr.shape
    shape_out = shape_in[:-2] + (6,)
    arr_out = np.zeros(shape_out, dtype=arr.dtype)
    for i, ij in imap_2d_to_1d.items():
        arr_out[..., i] = arr[..., ij[0], ij[1]]
    return arr_out


def _map_6x1_to_3x3(arr: ndarray, imap_2d_to_1d: dict = _imap_2d_to_1d) -> ndarray:
    shape_in = arr.shape
    shape_out = shape_in[:-1] + (3, 3)
    arr_out = np.zeros(shape_out, dtype=arr.dtype)
    for i, ij in imap_2d_to_1d.items():
        arr_out[..., ij[0], ij[1]] = arr[..., i]
        arr_out[..., ij[1], ij[0]] = arr[..., i]
    return arr_out


def _map_3x3x3x3_to_6x6(arr: ndarray, imap_4d_to_2d: dict = _imap_4d_to_2d) -> ndarray:
    shape_in = arr.shape
    shape_out = shape_in[:-4] + (6, 6)
    arr_out = np.zeros(shape_out, dtype=arr.dtype)
    for ij, ijkl in imap_4d_to_2d.items():
        arr_out[..., ij[0], ij[1]] = arr[..., ijkl[0], ijkl[1], ijkl[2], ijkl[3]]
    return arr_out


def _map_6x6_to_3x3x3x3(arr: ndarray, imap_4d_to_2d: dict = _imap_4d_to_2d) -> ndarray:
    shape_in = arr.shape
    shape_out = shape_in[:-2] + (3, 3, 3, 3)
    arr_out = np.zeros(shape_out, dtype=arr.dtype)
    for ij, ijkl in imap_4d_to_2d.items():
        arr_out[..., ijkl[0], ijkl[1], ijkl[2], ijkl[3]] = arr[..., ij[0], ij[1]]
        arr_out[..., ijkl[1], ijkl[0], ijkl[2], ijkl[3]] = arr[..., ij[0], ij[1]]
        arr_out[..., ijkl[0], ijkl[1], ijkl[3], ijkl[2]] = arr[..., ij[0], ij[1]]
        arr_out[..., ijkl[1], ijkl[0], ijkl[3], ijkl[2]] = arr[..., ij[0], ij[1]]
    return arr_out
