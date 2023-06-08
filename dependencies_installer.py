import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages
packages = [
    'pandas',
    'alpha_vantage',
    'matplotlib',
    'sklearn',
    'tensorflow',
    'seaborn',
    'numpy',
    'keras_tuner'
]

# Try to import the packages, if not installed, install them
for package in packages:
    try:
        dist = __import__(package)
        print(f"{package} is installed")
    except ImportError:
        print(f"{package} is NOT installed")
        install(package)

# 'alpha_vantage.timeseries' is part of the 'alpha_vantage' package
# 'matplotlib.pyplot' is part of the 'matplotlib' package
# 'pandas.DataFrame' is part of the 'pandas' package
# 'sklearn.preprocessing' is part of the 'sklearn' package
# 'tensorflow.keras' is part of the 'tensorflow' package

# For 'utils', you would need to have this python file or module in your python path or your script's directory.
# It's not something you can install with pip.
