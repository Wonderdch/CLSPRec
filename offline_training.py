import os
import sys

os.system("nohup sh -c '" + sys.executable + " main.py' & echo $!")
