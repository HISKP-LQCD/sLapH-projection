import os

################################################################################
# checks if the directory where the file will be written does exist
def ensure_dir(f):
#  d = os.path.dirname(f)
  if not os.path.exists(f):
    os.makedirs(f)

