import os
from utils import split_between_last_char
import time

def GenerateLabel(path):
    sanitize = "../orbit-counting/python/sanitize.py"
    orbit_counts = "../orbit-counting/wrappers/orbit_counts.py"
    outPath = "../orbit-counting/wrappers/"
    outName = "out.txt"
    prePath, fileName = split_between_last_char(path, '/')
    print(prePath, fileName)
    os.system('python ' + sanitize + " " + prePath + " " + fileName)
    prefix, _ = split_between_last_char(fileName, '.')
    suffix = 'edges'
    print(prefix, suffix)
    print('python3 ' + orbit_counts + " " + prePath + "/" + prefix + "." + suffix  + " 5 -c")
    os.system('python3 ' + orbit_counts + " " + prePath + "/" + prefix + "." + suffix  + " 5 -c")
    os.system("mv ./out.txt " + prefix + ".out")
    os.system("mv " + prefix + ".out " + prePath)
    
if __name__ == '__main__': 
    time_start = time.time()
    GenerateLabel("../newData/com-orkut.edge")
    time_end = time.time()
    print("end: time cost" + str(time_end - time_start) + "s")