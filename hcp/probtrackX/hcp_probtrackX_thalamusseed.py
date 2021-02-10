#!/usr/bin/env python

import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.freesurfer as fs
from nipype import Node
from nipype.interfaces.utility import IdentityInterface
import os

def hemispherize(in_files):
    lh_list = [x for x in in_files if "lh_" in x or "brainstem" in x]
    rh_list = [x for x in in_files if "rh_" in x or "brainstem" in x]
    return [lh_list, rh_list]

def pbX_wf(subject_id,
           sink_directory,
           name='hcp_pbX'):

    hcp_pbX_wf = pe.Workflow(name='hcp_pbX_wf')

    #making all the keys for the dictionary
    info = dict(merged_thsamples=[['subject_id', 'merged_th']],
                merged_phsamples=[['subject_id', 'merged_ph']],
                merged_fsamples=[['subject_id', 'merged_f']],
                dmri_brain=[['subject_id','T1w_acpc_dc_restore_1.25']],
                fs_brain=[['subject_id','T1w_acpc_dc']],
                aparcaseg=[['subject_id','aparc+aseg']],
                mask=[['subject_id','nodif_brain_mask']])

    # Create a datasource node to get the dwi, bvecs, and bvals
    #This uses the dictionary created above and inputs the keys from the dictionary
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],outfields=list(info.keys())),name = 'datasource')
    datasource.inputs.template = '%s/%s'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.base_directory = os.path.abspath('/home/data/hcp')
    datasource.inputs.field_template = dict(merged_thsamples='/home/data/madlab/data/mri/hcp/bedpostX/%s/hcpbpX/thsamples/%s*.nii.gz',
                                            merged_phsamples='/home/data/madlab/data/mri/hcp/bedpostX/%s/hcpbpX/phsamples/%s*.nii.gz',
                                            merged_fsamples='/home/data/madlab/data/mri/hcp/bedpostX/%s/hcpbpX/fsamples/%s*.nii.gz',
                                            dmri_brain='/home/data/hcp/%s/T1w/%s.nii.gz',
                                            fs_brain='/home/data/hcp/%s/T1w/%s.nii.gz',
                                            aparcaseg='/home/data/hcp/%s/T1w/%s.nii.gz',
                                            mask='/home/data/hcp/%s/T1w/Diffusion/%s.nii.gz')
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True

    # Create a flirt node to calculate the dmri_brain to fs_brain xfm
    #Basically creating a conversion from DWI space to Freesurfer space
    dmri2fs_xfm = pe.Node(fsl.FLIRT(), name = 'dmri2fs_xfm')
    dmri2fs_xfm.inputs.out_matrix_file = 'dmri_2_fs_xfm.mat'
    hcp_pbX_wf.connect(datasource, 'dmri_brain', dmri2fs_xfm, 'in_file')
    hcp_pbX_wf.connect(datasource, 'fs_brain', dmri2fs_xfm, 'reference')

    # Create a convertxfm node to create inverse xfm of dmri2fs affine
    # Basicaaly creating a conversion from freesurfer space to DWI space
    invt_dmri2fs = pe.Node(fsl.ConvertXFM(), name= 'invt_dmri2fs')
    invt_dmri2fs.inputs.invert_xfm = True
    invt_dmri2fs.inputs.out_file = 'fs_2_dmri_xfm.mat'
    hcp_pbX_wf.connect(dmri2fs_xfm, 'out_matrix_file', invt_dmri2fs, 'in_file')

    # Extract thalamus seed masks from aparc+aseg.nii.gz file
    # Here 10 is the left thalamus, and 49 is the right thalamus
    thal_seed_mask = pe.MapNode(fs.Binarize(),
                               iterfield=['match', 'binary_file'],
                               name='thal_seed_mask')
    #thal_seed_mask.inputs.subject_dir = 'aparcaseg'
    thal_seed_mask.inputs.match = [[10],[49]]
    thal_seed_mask.inputs.binary_file = ['lft_thal.nii.gz', 'rt_thal.nii.gz']
    hcp_pbX_wf.connect(datasource, 'aparcaseg', thal_seed_mask, 'in_file')
    
    #Next we need to avoid the ventricles by creating an -avoid_mask
    #There are no left and right 3rd and 4th ventricle, so we are making one mask
    avoid_mask = pe.Node(fs.Binarize(), 
                         #out_type='nii.gz', 
                         name='avoid_mask')
    #avoid_mask.inputs.subject_dir = 'aparcaseg'
    avoid_mask.inputs.match = [4, 14, 15, 43, 72] #lft_lat_ven, 3rd_ven, 4th_ven, rgt_lat_ven, 5th_ven
    avoid_mask.inputs.binary_file = 'ventricles.nii.gz'
    hcp_pbX_wf.connect(datasource, 'aparcaseg', avoid_mask, 'in_file')


    # Extract cortical target masks from aparc+aseg.nii.gz file
    # The ".match" is the freesurfer label and the binary_file is the label/name
    ctx_targ_mask = pe.MapNode(fs.Binarize(),
                               iterfield=['match', 'binary_file'],
                               name='ctx_targ_mask')
    #ctx_targ_mask.inputs.subject_dir = 'aparcaseg'
    ctx_targ_mask.inputs.match = [[1024], [1022], [1003, 1028, 1027, 1012, 1019, 1020, 1032],
                                  [1031, 1029, 1008], [1009, 1015, 1033, 1035, 1034, 1030],
                                  [1011], [1017], [1002], [1014], [1026], [1028],
                                  [1023, 1025, 1010], [1005, 1013, 1021], [1007], [1006], [1016], [17], [18], [26],
                                  [2024], [2022], [2003, 2028, 2027, 2012, 2019, 2020, 2032],
                                  [2031, 2029, 2008], [2009, 2015, 2033, 2035, 2034, 2030],
                                  [2011], [2017], [2002], [2014], [2026], [2028],
                                  [2023, 2025, 2010], [2005, 2013, 2021], [2007], [2006], [2016], [53], [54], [58]]
    ctx_targ_mask.inputs.binary_file = ['ctx_lh_precentral.nii.gz', 'ctx_lh_postcentral.nii.gz',
                                        'ctx_lh_latfront.nii.gz', 'ctx_lh_parietal.nii.gz', 'ctx_lh_temporal.nii.gz',
                                        'ctx_lh_occipital.nii.gz', 'ctx_lh_paracentral.nii.gz', 'ctx_lh_caudantcing.nii.gz',
                                        'ctx_lh_medorbfront.nii.gz', 'ctx_lh_rostantcing.nii.gz', 'ctx_lh_superfront.nii.gz',
                                        'ctx_lh_medpost.nii.gz', 'ctx_lh_medoccipital.nii.gz', 'ctx_lh_fusiform.nii.gz',
                                        'ctx_lh_entorhinal.nii.gz', 'ctx_lh_parahippocampal.nii.gz', 'lh_hpc.nii.gz', 'lh_amy.nii.gz', 'lh_nacc.nii.gz',
                                        'ctx_rh_precentral.nii.gz', 'ctx_rh_postcentral.nii.gz',
                                        'ctx_rh_latfront.nii.gz', 'ctx_rh_parietal.nii.gz', 'ctx_rh_temporal.nii.gz',
                                        'ctx_rh_occipital.nii.gz', 'ctx_rh_paracentral.nii.gz', 'ctx_rh_caudantcing.nii.gz',
                                        'ctx_rh_medorbfront.nii.gz', 'ctx_rh_rostantcing.nii.gz', 'ctx_rh_superfront.nii.gz',
                                        'ctx_rh_medpost.nii.gz', 'ctx_rh_medoccipital.nii.gz', 'ctx_rh_fusiform.nii.gz',
                                        'ctx_rh_entorhinal.nii.gz', 'ctx_rh_parahippocampal.nii.gz', 'rh_hpc.nii.gz', 'rh_amy.nii.gz', 'rh_nacc.nii.gz']
    hcp_pbX_wf.connect(datasource, 'aparcaseg', ctx_targ_mask, 'in_file')



    # Create a flirt node to apply inverse transform to seeds
    # Basically you convert the masks (seeds) that were in freesurfer space to the DWI space
    seedxfm_fs2dmri = pe.MapNode(fsl.FLIRT(),
                                 iterfield = ['in_file'],
                                 name='seedxfm_fs2dmri')
    seedxfm_fs2dmri.inputs.apply_xfm = True
    seedxfm_fs2dmri.inputs.interp = 'nearestneighbour'
    hcp_pbX_wf.connect(thal_seed_mask, 'binary_file', seedxfm_fs2dmri, 'in_file')
    hcp_pbX_wf.connect(datasource, 'dmri_brain', seedxfm_fs2dmri, 'reference')
    hcp_pbX_wf.connect(invt_dmri2fs, 'out_file', seedxfm_fs2dmri, 'in_matrix_file')

    # Create a flirt node to apply inverse transform to targets
    # You do the same as the previous node, but to the target masks
    targxfm_fs2dmri = pe.MapNode(fsl.FLIRT(),
                                 iterfield = ['in_file'],
                                 name='targxfm_fs2dmri')
    targxfm_fs2dmri.inputs.apply_xfm = True
    targxfm_fs2dmri.inputs.interp = 'nearestneighbour'
    hcp_pbX_wf.connect(ctx_targ_mask, 'binary_file', targxfm_fs2dmri, 'in_file')
    hcp_pbX_wf.connect(datasource, 'dmri_brain', targxfm_fs2dmri, 'reference')
    hcp_pbX_wf.connect(invt_dmri2fs, 'out_file', targxfm_fs2dmri, 'in_matrix_file')

    #Apply the inverse transform for the avoid masks from freesurfer to DWI space
    avoidmaskxfm_fs2dmri = pe.Node(fsl.FLIRT(),
                                   name='avoidmaskxfm_fs2dmri')
    avoidmaskxfm_fs2dmri.inputs.apply_xfm = True
    avoidmaskxfm_fs2dmri.inputs.interp = 'nearestneighbour'
    hcp_pbX_wf.connect(avoid_mask, 'binary_file', avoidmaskxfm_fs2dmri, 'in_file')
    hcp_pbX_wf.connect(datasource, 'dmri_brain', avoidmaskxfm_fs2dmri, 'reference')
    hcp_pbX_wf.connect(invt_dmri2fs, 'out_file', avoidmaskxfm_fs2dmri, 'in_matrix_file')

    # Compute motion regressors (save file with 1st and 2nd derivatives)
    #make_targ_lists = pe.Node(util.Function(input_names=['in_files'],
    #                                        output_names='out_list',
    #                                        function=create_two_lists),
    #                          name='make_targ_lists')
    #hcp_pbX_wf.connect(targxfm_fs2dmri, 'out_file', make_targ_lists, 'in_files')

    #PROBTRACKX NODE 
    pbx2 = pe.MapNode(fsl.ProbTrackX2(),
                      iterfield = ['seed', 'target_masks'], #Should I have included avoid_mp here?
                      name='pbx2')
    pbx2.inputs.c_thresh = 0.2
    pbx2.inputs.n_steps=2000
    pbx2.inputs.step_length=0.5
    pbx2.inputs.n_samples=25000
    pbx2.inputs.opd=True
    pbx2.inputs.os2t=True
    pbx2.inputs.loop_check=True
    #pbx2.plugin_args = {'bsub_args': '-q PQ_madlab'} #old way new way below
    pbx2.plugin_args = {'sbatch_args': ('-p IB_40C_1.5T --qos pq_madlab --account iacc_madlab -N 1 -n 6')}
    hcp_pbX_wf.connect(datasource, 'merged_thsamples', pbx2, 'thsamples')
    hcp_pbX_wf.connect(datasource, 'merged_phsamples', pbx2, 'phsamples')
    hcp_pbX_wf.connect(datasource, 'merged_fsamples', pbx2, 'fsamples')
    hcp_pbX_wf.connect(seedxfm_fs2dmri, 'out_file', pbx2, 'seed')
    hcp_pbX_wf.connect(targxfm_fs2dmri, ('out_file', hemispherize), pbx2, 'target_masks')
    #hcp_pbX_wf.connect(make_targ_lists, 'out_list', pbx2, 'target_masks')
    hcp_pbX_wf.connect(avoidmaskxfm_fs2dmri, 'out_file', pbx2, 'avoid_mp')
    hcp_pbX_wf.connect(datasource, 'mask', pbx2, 'mask')
    

    # Create a findthebiggest node to do hard segmentation between
    # seeds and targets
    #basically this segments the seed region on the basis of outputs of probtrackX when classification targets are being used. 
    findthebiggest = pe.MapNode(fsl.FindTheBiggest(),
                                iterfield = ['in_files'],
                                name='findthebiggest')
    hcp_pbX_wf.connect(pbx2, 'targets', findthebiggest, 'in_files')

    # Create a datasink node to save outputs.
    datasink = pe.Node(interface=nio.DataSink(),name='datasink')
    datasink.inputs.base_directory = os.path.abspath(sink_directory)
    datasink.inputs.container = subject_id + '/' + 'thal_seed'
    hcp_pbX_wf.connect(pbx2, 'log', datasink, 'hcpprobX.log')
    hcp_pbX_wf.connect(pbx2, 'fdt_paths', datasink, 'hcpprobX.fdt')
    hcp_pbX_wf.connect(pbx2, 'way_total', datasink, 'hcpprobX.waytotal')
    hcp_pbX_wf.connect(pbx2, 'targets', datasink, 'hcpprobX.targets')
    hcp_pbX_wf.connect(findthebiggest, 'out_file', datasink, 'hcpprobX.fbiggest.@biggestsegmentation')
    #hcp_pbX_wf.connect(thal_seed_mask, 'binary_file', datasink, 'hcpprobX.thal_mask')
    hcp_pbX_wf.connect(seedxfm_fs2dmri, 'out_file', datasink, 'hcpprobX.seed_masks')
    #from seed_xsfm(out_file) to datasink "seed_files"
    #do we need this - > emu_pbX_wf.connect(datasource, 'ref_b0', datasink, 'emuprobX.b0')
    #do we need this - > emu_pbX_wf.connect(thal_seed_mask, 'binary_file', datasink, 'emuprobX.thal_mask')

    return hcp_pbX_wf


"""
Creates the full workflow
"""

def create_probX_workflow(args, name='hcp_probX'):
    
    kwargs = dict(subject_id=args.subject_id,
                  sink_directory=os.path.abspath(args.out_dir),
                  name=name)
    probX_workflow = pbX_wf(**kwargs)
    return probX_workflow

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--subject_id", dest="subject_id",
                        help="Current subject id", required=True)
    parser.add_argument("-o", "--output_dir", dest="out_dir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Working directory base")
    args = parser.parse_args()

    wf = create_probX_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()
    
    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/hcp_probX'
    wf.base_dir = work_dir + '/' + args.subject_id
    # OLD: wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_madlab'})
    wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p IB_44C_512G --qos pq_madlab --account iacc_madlab -N 1 -n 1'), 'overwrite': True})
