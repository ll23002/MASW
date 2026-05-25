import sys, os, warnings

warnings.filterwarnings("ignore")

CPS_BIN = "/home/alexander/CPS/PROGRAMS.330/bin/"
BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)

DX = 2.0
N_TRACES = 24

F_MIN = 0.1
F_MAX = 30.0

# 1/0.002 = 500Hz
DT = 0.002
# 8192 * 0.002 = 16.38S
# 1/16.38 = 0.061s
N_SAMPLES = 8192

N_LAYER = 5
