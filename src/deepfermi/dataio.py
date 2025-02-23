import os
import re


def get_files(pname, inc_strg=[], ex_strg=[], vis=0):
    """
    Retrieves a list of files from a specified directory while filtering based
    on inclusion and exclusion patterns.

    Parameters:
        pname (str): The path to the directory from which files are to be
                     retrieved.
        inc_strg (list of str: A list of substrings. Only files
                               whose names contain any of these substrings
                               will be included in the result. Default is
                               an empty list, meaning no filtering.
        ex_strg (list of str): A list of substrings. Files whose names
                               contain any of these substrings will be
                               excluded from the result. Default is an empty
                               list, meaning no filtering.
        vis (int): A flag (0 or 1) indicating whether to print the filtered
                   list of files. Default is 0 (do not print).

    Returns:
        list of str: A list of filenames in the specified directory that match
                     the filtering criteria.

    """

    # Get all files in directory but ignore files starting with .
    reg = re.compile('^\\..*')
    flist = [x for x in sorted(os.listdir(pname)) if not reg.match(x)]
    
    # Exclude files based on ex_strg
    for ind in range(len(ex_strg)):
        reg = re.compile('.*' + ex_strg[ind] + '.*')
        flist = [x for x in flist if not reg.match(x)]
        
    # Include only files based on inc_strg
    for ind in range(len(inc_strg)):
        reg = re.compile('.*' + inc_strg[ind] + '.*')
        flist = [x for x in flist if reg.match(x)]

    # List found files
    if vis:
        for ind in range(len(flist)):
            print('%d:%s' % (ind, flist[ind]))

    return (flist)

