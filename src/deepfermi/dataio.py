import os, re

def get_files(pname, inc_strg=[], ex_strg=[], vis=0):
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
            
    return(flist)
    
