from dewloosh.core.downloads import download_stand, _download_file
from polymesh import PolyData
from typing import Union


def download_bernoulli_console_json_linstat():  # pragma: no cover
    """
    Downloads the description of a simple bernoulli console as a json file.

    Returns
    -------
    str
        A path to a file on your filesystem.

    Example
    --------
    >>> from sigmaepsilon.examples import download_bernoulli_console_json_20B2
    >>> jsonpath = download_bernoulli_console_json_20B2()

    """
    return _download_file("console_bernoulli_linstat.json")[0]


"""def stand_vtk(read=False) -> Union[str, PolyData]:
    vtkpath = download_stand()
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath"""
