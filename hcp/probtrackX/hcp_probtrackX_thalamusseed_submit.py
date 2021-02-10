#!/usr/bin/env python

#SBATCH --job-name=hcp_pbx_atm
#SBATCH --partition IB_16C_96G
#SBATCH --account iacc_madlab
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/hcp_err
#SBATCH -o /scratch/madlab/crash/hcp_out

import os

#subject_list = ['102816']
#'102311', '101309','101410']

subject_list = ['101309', '101410','101915', '101006', '101107', '100307', '108323', '119833', '138231','148335']
#subject_list = ['102008', '103111', '100408', '103414', '105115', '105216', '106319', '108121', '106521', '107422']
#subject_list = ['107321', '111716', '113619', '110411', '109123', '111312', '108525', '113821', '113215', '111413']
#subject_list = ['114419', '113922', '112819', '108828', '120212', '122317',  '118932', '121618', '122620', '120111']
#subject_list = ['123420', '124220', '123925', '128127', '126628', '128632', '126325', '127933', '124422', '124826']
#subject_list = ['125525', '127630', '123117', '118730', '115320', '117122', '118528', '116524', '117324', '120515']
#subject_list = ['131217', '130922', '131722', '130316', '131924', '129028', '137128', '130013', '137027', '136833']
#subject_list = ['133827', '133625', '135528', '136227', '133928', '133019', '132118', '134324', '135932', '135225', '158136']
#subject_list = ['138534', '139233', '137633', '141826', '142626', '142828', '139637', '144226', '143325', '140420', '156233']
#subject_list = ['140925', '141422', '140824', '140117', '145834', '147737', '146432', '144832', '146331', '147030', '158540']
#subject_list = ['148941', '152831', '151627', '150726', '149337', '150625', '148032', '148840', '149539', '149741', '156637']
#subject_list = ['150423', '151223', '151728', '150524', '154734', '151526', '153429', '153833', '153025', '154431', '158035']
#subject_list = ['154835', '155231', '155635', '154936', '157437', '159340', '159441', '159239', '157336', '160123', '159138']


workdir = '/scratch/madlab/hcp_probX_PR'
outdir = '/home/data/madlab/data/mri/hcp/probtrack/thalamus_25ksamp_avoidmask_january'
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
