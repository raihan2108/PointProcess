""" 
Experiment User_Level_Yearly:
- Number of Users are 5000
- This experiment captures user behavior change yearly.
- The input data is stored at 'experiment_input_data/5000_users/user_batches/yearly/input'
- The model parameters are stored at experiment_input_data/5000_users/user_batches/monthly/output


Experiment User_Level_Monthly:
    TODO

"""
import pandas as pd
import glob
import os
import sys
import json
import gzip

def read_input_data(inputfile,granularity="monthly"):
    """ 
        This method is used to read data from input file, break values down into smaller chunks and write to output files.
    """

def _yearly_helper(inputfile,_unit='yearly',entity='user',_vals=[2015,2016]):
    users=list()
    outputdir=os.path.abspath(os.path.join(inputfile,os.path.pardir,os.path.pardir))+"/"+entity+"_batches/"+_unit+"/"

    with gzip.open(inputfile,"rb") as f:
        for line in f:
            users.append(json.loads(line.decode('utf-8')))
    if _unit=='yearly':
        unit='year'
    elif _unit=='monthly':
        unit='month'
    elif _unit=='weekly':
        unit='week'
    else:
        raise Exception("_unit has to be one of {yearly,monthly,weekly}")
    vals=_vals
    for val in _vals:
        filtered_obj=list()
        outputpath=outputdir+str(val)+"/input/"+os.path.basename(inputfile)
        print("Outputfile",outputpath)
        for user in users:
            ser=pd.to_datetime(pd.Series(user['arrivalTimes']))
            indices=ser.apply(lambda x:x if getattr(x,unit)==val else None).dropna().index
            odpairs=[user['ODpairs'][i] for i in indices]
            delays=[user['delays'][i] for i in indices]
            arrivaltimes=[user['arrivalTimes'][i] for i in indices]
            _id=user['_id']
            obj={'ODpairs':odpairs,'delays':delays,'arrivalTimes':arrivaltimes,'_id':_id}
            filtered_obj.append(obj)
        #write to file
        with open(outputpath, 'w') as g:
            for obj in filtered_obj:
                g.write(json.dumps(obj)+'\n')

    inF = open(outputpath, 'rb')
    s = inF.read()
    inF.close()

    outF = gzip.GzipFile(outputpath, 'wb')
    outF.write(s)
    outF.close()

def yearly_user(inputdir):
    for item in glob.glob(inputdir+"/*.gz"):
         _yearly_helper(item)


def create_input_files_for_experiments(inputdir,temp_gran='yearly',entity='user'):
    """
        We will run three types of experiments
            Temporal Granularity
                1. Weekly
                2. Monthly
                3. Yearly

            Entity Unit (refers to the smallest entity in dataset)
                1. User Level
                2. Station Level
                3. Region Level

    """
    if temp_gran=='yearly':
        if entity=='user':
           yearly_user(inputdir)
        elif entity=='station':
            pass
        elif entity=='region':
            pass
        else:
            raise Exception('entity has to be one of {user,station,region}')
    elif temp_gran=='monthly':
        if entity=='user':
            pass
        elif entity=='station':
            pass
        elif entity=='region':
            pass
        else:
            raise Exception('entity has to be one of {user,station,region}')
    elif temp_gran=='weekly':
        if entity=='user':
                pass
        elif entity=='station':
            pass
        elif entity=='region':
            pass
        else:
            raise Exception('entity has to be one of {user,station,region}')
    else:
        raise Exception('temporal granularity has to be one of {yearly,monthly,weekly}')

if __name__=="__main__":
    create_input_files_for_experiments(inputdir=sys.argv[1],temp_gran=sys.argv[2],entity=sys.argv[3])
