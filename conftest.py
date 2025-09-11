import pytest
import platform
import numpy as np
import pandas as pd
import os

@pytest.fixture
def rtol():
    return 1e-3

@pytest.fixture
def atol():
    return 1e-6

@pytest.fixture
def os_name(request):

    # find the operating system name
    system = platform.system()
    filename = ""
    if system == "Windows":
        sysname = "windows_2022"
    elif system == "Darwin":
        sysname = "macos_15"
    else:
        pytest.skip(f"No benchmark for current OS '{system}'")

    return sysname
