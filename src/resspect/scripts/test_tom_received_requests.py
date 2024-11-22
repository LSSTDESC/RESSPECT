import os
from resspect.tom_client import TomClient

#pulls recommended spec by resspect from the TOM
def which_spec(url, username, passwordfile=None, password=None, requested_since=None):
    tom = TomClient(url = url, username = username, password = password, 
                    passwordfile = passwordfile)
    dic = {'detected_since_mjd':0}
    if requested_since is not None:
        dic['requested_since'] = requested_since

    res = tom.post( 'elasticc2/spectrawanted', json=dic )

    assert res.status_code == 200
    assert res.json()['status'] == "ok"
    reqs = res.json()
    return reqs

def contains_correct_requests(url, username, passwordfile):
    req = which_spec(url, username, passwordfile)
    return(len(req['wantedspectra']) == batch-1)
    
url = os.environ.get('TOM_URL', "https://desc-tom-2.lbl.gov")
username = os.environ.get('TOM_USERNAME', None)
passwordfile = os.environ.get('TOM_PASSWORDFILE', None)
batch = os.environ.get('RESSPECT_BATCH', 5 )

# check that we have the same number of requests as we sent
if contains_correct_requests(url, username, passwordfile):
    exit(0)
else:
    exit(1)