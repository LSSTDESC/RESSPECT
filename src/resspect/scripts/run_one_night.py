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
from pathlib import Path
import numpy as np

url = os.environ.get('TOM_URL', "https://desc-tom-2.lbl.gov")
day = os.environ.get('RESSPECT_DAY', 5 )
username = os.environ.get('TOM_USERNAME', None)
password = os.environ.get('TOM_PASSWORDFILE', None)
batch = os.environ.get('RESSPECT_BATCH', 5)
outdir = Path(os.environ.get('RESSPECT_OUTDIR', "TOM_days_storage")).resolve()

def get_one_night_data(day, url, username, passwordfile):

    """
    Get hot light curves from TOM for one night
    """

    data_dic = request_TOM_data(url = url,username = username,
                                passwordfile = passwordfile, detected_in_last_days = 1, 
                                mjdnow = 60406+day, 
                                #cheat_gentypes = [82, 10, 21, 27, 26, 37, 32, 36, 31, 89]
                                )
    data_dic=data_dic['diaobject']
    
    #get features from that data
    file_name = outdir / ('TOM_hot_features_day_'+str(day)+'.csv')
        
    fit_TOM(data_dic, output_features_file = file_name, feature_extractor = 'Malanchev')
    data = pd.read_csv(file_name, index_col=False)
    data['orig_sample'] = 'pool'
    
    data.to_csv(file_name,index=False)

def run_one_night(day, url: str = "https://desc-tom-2.lbl.gov", username: str = 'amandaw8', 
                  passwordfile: str = '/Users/arw/secrets/tom2', batch: int = 5):
    
    """
    Runs RESSPECT for one night

    Parameters
    ----------
    day
        day of the survey
    url
        URL of the TOM
    username
        username for the TOM
    passwordfile
        path to the password file
    batch
        number of recommended supernovae for the night
    """
     
    #get new lc info from TOM (from yesterday (for now))
    get_one_night_data(day-1, url, username, passwordfile)

    # -------------------------

    #get new lc info from TOM (for today)
    get_one_night_data(day, url, username, passwordfile)
    
    
    # run the loop to get queried objects and updated metrics
    days = [day-1, day+1]                                # first and last day of the survey
    
    strategy = 'UncSampling'                        # learning strategy
    batch = batch                                       # if int, ignore cost per observation,
                                                         # if None find optimal batch size
    
    sep_files = True                               # if True, expects train, test and
                                                        # validation samples in separate filess
    
    path_to_features_dir = 'TOM_days_storage/'   # folder where the files for each day are stored
    
    # output results for metrics
    output_metrics_file = 'results/metrics_' + strategy + '_' + \
                           '_batch' + str(batch) +  '.csv'
    
    # output query sample
    output_query_file = 'results/queried_' + strategy + '_'  + \
                            '_batch' + str(batch) + '_day_'+ str(day) + '.csv'
    
    path_to_ini_files = {}
    
    # features from full light curves for initial training sample
    path_to_ini_files['train'] = str(outdir / 'TOM_training_features')
    path_to_ini_files['test'] = str(outdir / 'TOM_test_features')
    path_to_ini_files['validation'] = str(outdir / 'TOM_validation_features')
    
    survey='LSST'
    
    classifier = 'RandomForest'
    n_estimators = 1000                             # number of trees in the forest
    
    feature_extraction_method = 'Malanchev'
    screen = False                                  # if True will print many things for debuging
    fname_pattern = ['TOM_hot_features_day_', '.csv']   # pattern on filename where different days
                                                        # are stored
    
    queryable= False                                 # if True, check brightness before considering
                                                        # an object queryable    
    
    # run time domain loop
    time_domain_loop(TimeDomainConfiguration(days=days, output_metrics_file=output_metrics_file,
                     output_queried_file=output_query_file,
                     path_to_ini_files=path_to_ini_files,
                     path_to_features_dir=path_to_features_dir,
                     strategy=strategy, fname_pattern=fname_pattern, batch=batch,
                     classifier=classifier,
                     sep_files=sep_files,
                     survey=survey, queryable=queryable,
                     feature_extraction_method=feature_extraction_method), 
                     screen=screen, n_estimators=n_estimators)
    # Read in RESSPECT requests to input to TOM format and find priorities
    ids = list(pd.read_csv(output_query_file)['id'])
    ids = [int(id) for id in ids]
    num = int(len(ids)/5)
    mod = len(ids)%5
    num_list = [num]*5
    mod_list = []
    for i in range(mod):
        mod_list.append(1)
    rem = 5-len(mod_list)
    mod_list = mod_list+[0]*rem
    num_list=list(np.asarray(num_list)+mod_list)
    priorities = []
    priorities.append([1]*num_list[0]+[2]*num_list[1]+[3]*num_list[2]+[4]*num_list[3]+[5]*num_list[4])    
    priorities = priorities[0]

    print(ids)
    print(priorities)
    
    # send these queried objects to the TOM
    submit_queries_to_TOM(url, username, passwordfile, objectids = ids, 
                          priorities = priorities, requester = 'resspect')
    

if __name__ == "__main__":
    run_one_night(day, url, username, password, batch)