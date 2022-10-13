#!/usr/bin/env python

#SBATCH --job-name=hcp_pbx_atm
#SBATCH --partition IB_16C_96G
#SBATCH --account iacc_madlab
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/hcp_err
#SBATCH -o /scratch/madlab/crash/hcp_out

import os

#Write down your subject in pyhton list format
subject_list = ['101309', '101410','101915', '101006', '101107', '100307', '108323', '119833', '138231','148335']


#Create a work directory where your intermediate files will be saved
workdir = '/scratch/madlab/hcp_probX_PR'

#create a output directory where your final probabilistic tractography results will be saved
outdir = '/home/data/madlab/data/mri/hcp/probtrack/thalamus_25ksamp_avoidmask_january'

#Creating a forloop that iterates over each subject and runs the subject through the main probabilistic script
for sid in subject_list:
    convertcmd = ' '.join(['python', 'hcp_probtrackX_thalamusseed.py', '-s', sid, '-o', outdir, '-w', workdir])
    script_file = 'hcp_pbx_atm-%s.sh' %sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n'])
        fp.writelines(['#SBATCH --job-name=hcp_pbx_atm_%s\n'%sid])
        fp.writelines(['#SBATCH --partition IB_44C_512G\n'])
        fp.writelines(['#SBATCH --account iacc_madlab\n'])
        fp.writelines(['#SBATCH --qos pq_madlab\n'])
        fp.writelines(['#SBATCH -e /scratch/madlab/crash/hcp_pbx_err_%s\n'%sid])
        fp.writelines(['#SBATCH -o /scratch/madlab/crash/hcp_pbx_out_%s\n'%sid])
        fp.writelines([convertcmd])
    outcmd = 'sbatch %s'%script_file
    os.system(outcmd)
    continue
