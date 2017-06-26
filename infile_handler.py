import argparse
import ConfigParser

def get_parameters():

  ##############################################################################
  # Argument parsing ########################################################### 

  parser = argparse.ArgumentParser()
  # parse the name of the infile and load its contents into the parser
  parser.add_argument("infile", help="name of input file")
  # verbosity is also parsed
  parser.add_argument("-v", "--verbose", action="store_true", \
                                                 help="increase output verbosity")
  # TODO: Only allow certain options, specify standard behavior, etc.
  parser.add_argument("-b", "--basis", choices=['cartesian', 'cyclic', \
                        'cyclic-i', 'cyclic-christian'], \
                        default='cyclic-christian',
                        help="continuum basis to be used in the program")
  
  args = parser.parse_args()

  ##############################################################################
  # Reading infile #############################################################
  
  config = ConfigParser.RawConfigParser()
  
  if(config.read(args.infile) == []):
    print "Error! Could not open infile: ", args.infile
    exit(-1)

  return args, config
  
 
