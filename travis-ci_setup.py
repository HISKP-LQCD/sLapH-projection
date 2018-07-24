# Taken from stackoverflow and modified
# https://stackoverflow.com/questions/4028697/how-do-i-download-a-zip-file-in-python-using-urllib2

import os
from urllib2 import urlopen, URLError, HTTPError


def dlfile(url):
    # Open the url
    try:
        f = urlopen(url)
        print "downloading " + url

        # Open our local file for writing
        with open(os.path.expanduser("~") + "/Data/" + os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())

    #handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


def main():
    url = "https://www.itkp.uni-bonn.de/~werner/sLapH-projection_integration-test_data/A40.24-cnfg0714.tar" 
    dlfile(url)

if __name__ == '__main__':
    main()
