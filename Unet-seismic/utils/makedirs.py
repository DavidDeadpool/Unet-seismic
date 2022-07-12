import os

def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        return