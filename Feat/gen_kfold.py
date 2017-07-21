
"""
__file__

    gen_kfold.py

__description__

    This file generates the StratifiedKFold indices which will be kept fixed in
    ALL the following model building parts.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import sys
import cPickle
from sklearn.cross_validation import StratifiedKFold
sys.path.append("../")
from param_config import config
import subprocess, os, sys


if __name__ == "__main__":


#   logname ="./log_output/" + sys.argv[0] + "_log.txt"    
# Unbuffer output
    '''
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    
    tee = subprocess.Popen(["tee", logname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    '''

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)

    skf = [0]*config.n_runs
    for stratified_label,key in zip(["duplicate"], ["is_duplicate"]):
     
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run+1)
            skf[run] = StratifiedKFold(dfTrain[key], n_folds=config.n_folds,shuffle=True, random_state=random_seed)
            for fold, (trainInd, validInd) in enumerate(skf[run]):
                print("================================")
                print("Index for run: %s, fold: %s" % (run+1, fold+1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, stratified_label), "wb") as f:
            cPickle.dump(skf, f, -1)

#    os.spawnve("P_WAIT", "/bin/ls", ["/bin/ls"], {})
#    os.execve("/bin/ls", ["/bin/ls"], os.environ)


