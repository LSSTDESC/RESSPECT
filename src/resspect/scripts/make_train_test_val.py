# This script was used to generate the TOM_*_features files in example_work_dir/TOM_days_storage
# that serve as the initial training data for run_one_night.py in github actions

import resspect
import pandas as pd
from resspect import request_TOM_data
from resspect import fit_TOM
from resspect import submit_queries_to_TOM
from resspect import time_domain_loop
from resspect.tom_client import TomClient
from resspect import time_domain_loop
from resspect import TimeDomainConfiguration
import os
import re
import numpy as np

def get_phot(obj_df):
    # get all of the photometry at once
    ids = obj_df['diaobject_id'].tolist()
    res = tom.post('db/runsqlquery/',
                          json={ 'query': 'SELECT diaobject_id, filtername, midpointtai, psflux, psfluxerr'  
                                ' FROM elasticc2_ppdbdiaforcedsource' 
                              ' WHERE diaobject_id IN (%s) ORDER BY diaobject_id, filtername, midpointtai;' % (', '.join(str(id) for id in ids)),
                                'subdict': {} } )
    all_phot = res.json()['rows']
    all_phot_df = pd.DataFrame(all_phot)
    # if you need mag from the arbitrary flux
    # all_phot_df['mag'] = -2.5*np.log10(all_phot_df['psflux']) + 27.5
    # all_phot_df['magerr'] = 2.5/np.log(10) * all_phot_df['psfluxerr']/all_phot_df['psflux']

    # format into a list of dicts
    data = []
    for idx, obj in obj_df.iterrows():
        phot = all_phot_df[all_phot_df['diaobject_id'] == obj['diaobject_id']]
        
        phot_d = {}
        phot_d['objectid'] = int(obj['diaobject_id'])
        phot_d['sncode'] = int(obj['gentype'])
        phot_d['redshift'] = obj['zcmb']
        phot_d['ra'] = obj['ra']
        phot_d['dec'] = obj['dec']
        phot_d['photometry'] = phot[['filtername', 'midpointtai', 'psflux', 'psfluxerr']].to_dict(orient='list')

        phot_d['photometry']['band'] = phot_d['photometry']['filtername']
        phot_d['photometry']['mjd'] = phot_d['photometry']['midpointtai']
        phot_d['photometry']['flux'] = phot_d['photometry']['psflux']
        phot_d['photometry']['fluxerr'] = phot_d['photometry']['psfluxerr']
        del phot_d['photometry']['filtername']
        del phot_d['photometry']['midpointtai']
        del phot_d['photometry']['psflux']
        del phot_d['photometry']['psfluxerr']
        
        data.append(phot_d)

    return data

url = os.environ.get('TOM_URL', "https://desc-tom-2.lbl.gov")
username = os.environ.get('TOM_USERNAME', None)
passwordfile = os.environ.get('TOM_PASSWORDFILE', None)

#MAKE INITIAL TRAINING SET 
objs = []

tom = TomClient(url = url, username = username, passwordfile = passwordfile)

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=10 limit 10;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=21 limit 5;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=31 limit 5;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

print("Printing objects from database...")
print(objs)

training_objs = get_phot(pd.DataFrame(objs))
outdir = 'TOM_days_storage'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fit_TOM(training_objs, output_features_file = outdir+'/TOM_training_features', feature_extractor = 'Malanchev')
data = pd.read_csv('TOM_days_storage/TOM_training_features',index_col=False)
data['orig_sample'] = 'train'
data["type"] = np.where(data["code"] == 10, 'Ia', 'other')
data.to_csv('TOM_days_storage/TOM_training_features',index=False)

#MAKE TEST SET 
objs = []

tom = TomClient(url = url, username = username, passwordfile = passwordfile)

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=10 limit 1000;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=21 limit 500;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=31 limit 500;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

test_objs = get_phot(pd.DataFrame(objs))
outdir = 'TOM_days_storage'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fit_TOM(test_objs, output_features_file = outdir+'/TOM_test_features', feature_extractor = 'Malanchev')
data = pd.read_csv('TOM_days_storage/TOM_test_features',index_col=False)
data['orig_sample'] = 'test'
data["type"] = np.where(data["code"] == 10, 'Ia', 'other')
data.to_csv('TOM_days_storage/TOM_test_features',index=False)

#MAKE VALIDATION SET 
objs = []

tom = TomClient(url = url, username = username, passwordfile = passwordfile)

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=10 limit 1000;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=21 limit 500;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

res = tom.post('db/runsqlquery/',
                        json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                              ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=31 limit 500;', 
                             'subdict': {}} )
objs.extend(res.json()['rows'])

val_objs = get_phot(pd.DataFrame(objs))
outdir = 'TOM_days_storage'

if not os.path.exists(outdir):
    os.makedirs(outdir)

fit_TOM(val_objs, output_features_file = outdir+'/TOM_validation_features', feature_extractor = 'Malanchev')
data = pd.read_csv('TOM_days_storage/TOM_validation_features',index_col=False)
data['orig_sample'] = 'validation'
data["type"] = np.where(data["code"] == 10, 'Ia', 'other')
data.to_csv('TOM_days_storage/TOM_validation_features',index=False)