#!/usr/bin/env python

#SBATCH --partition IB_16C_96G
#SBATCH --qos pq_madlab
#SBATCH --account iacc_madlab
#SBATCH -e /scratch/madlab/puck/hcp_kmeans_5controls/crash/hcp_kmeans_err
#SBATCH -o /scratch/madlab/puck/hcp_kmeans_5controls/crash/hcp_kmeans_out

import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface

n_clus = 8
def kmeans_func(subject_id, n_clus, thal_masks):

    import os
    import numpy as np
    import nibabel as nib
    import scipy as sp
    import pandas as pd
    from sklearn.cluster import KMeans
    from glob import glob
    root_dir = '/home/data/madlab/data/mri/hcp/probtrack/thalamus_25ksamp_avoidmask_january' #probtrackX output
    out_files = []
    target_clusters = []
    out_features = []

    #Here it is iterating over the left and right hemispheres so we can analyze the left and right
    #hemispheres separately. Later this will be combined
    for hemi in ['0', '1']:
        if hemi == '0':
            out_file = os.path.join(os.getcwd(), 'kmeans_thallh_{0}-clus.nii.gz'.format(n_clus))
            curr_thal_mask = thal_masks[0]
        else:
            out_file = os.path.join(os.getcwd(), 'kmeans_thalrh_{0}-clus.nii.gz'.format(n_clus))
            curr_thal_mask = thal_masks[1]

        # The two following lines will give me a list of the targets for the current hemisphere
        data_dir = root_dir + '/' + subject_id + '/thal_seed/hcpprobX/targets/_pbx2' + hemi + '/*.nii.gz'
        targ_files = sorted(glob(data_dir))

        # Use the mask image to get some information about the dimensions of the data
        # Affine, header, etc... for later saving and reshaping
        hemi_mask_img = nib.load(curr_thal_mask)
        hemi_mask_data_affine = hemi_mask_img.affine
        hemi_mask_data_header = hemi_mask_img.header
        hemi_mask_data = hemi_mask_img.get_data()
        hemi_mask_data_dims = hemi_mask_data.shape

        # Replace posterior slices of mask with 0 to not analyze
        # voxels that are physically adjacent to the hippocampus
        # these voxels will have artifically inflated target hits.
        hemi_mask_data[:,0:80,:] = 0

        # Iterate over the target files
        for i, file_name in enumerate(targ_files):
            # Load the current target file and get the data
            curr_targ_img = nib.load(file_name)
            curr_targ_affine = curr_targ_img.affine
            curr_targ_header = curr_targ_img.header
            curr_targ_img_data = curr_targ_img.get_data()

            #Hastagged out bc its already up top, and will be used below
            # Replace posterior slices with 0
            #curr_targ_img_data[:,0:79,:] = 0

            # Mask the curr_targ_img_data by the current hemisphere mask
            curr_targ_img_data_onlythal = curr_targ_img_data[hemi_mask_data > 0]

            # Put the masked thalamus data into a n-by-y  matrix
            # n = number of voxels
            # y = number of features (i.e., connections to targets)
            if i == 0:
                input_array = curr_targ_img_data_onlythal
            else:
                input_array = np.column_stack((input_array, curr_targ_img_data_onlythal))

        # Run the kmeans algorithm
        thalamus_kmeans = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
        #adds 1 to the cluster assignment to prevent 0 cluster from fading
        #in to non-thalamus volume
        thal_kmeans_out = thalamus_kmeans.fit_predict(input_array) + 1

        #amyg_kmeans_results = amyg_kmeans_out.reshape(hemi_dims[0] * hemi_dims[1] * hemi_dims[2], 1)
        thal_kmeans_results = hemi_mask_data.copy().reshape(hemi_mask_data_dims[0] * hemi_mask_data_dims[1] * hemi_mask_data_dims[2], 1)

        counter = 0
        for idx, value in enumerate(thal_kmeans_results):
            if value > 0:
                thal_kmeans_results[idx] = thal_kmeans_out[counter]
                counter += 1
        # Reshape the column label data back into the original n x y x n shape
        thal_kmeans_results = thal_kmeans_results.reshape(hemi_mask_data_dims)

        # Save the newly labeled thalamus voxels as a .nii.gz file
        thal_kmeans_img = nib.Nifti1Image(thal_kmeans_results, curr_targ_affine, header=curr_targ_header)
        thal_kmeans_img.to_filename(out_file)
        out_files.append(out_file)

        for x in range(len(thalamus_kmeans.cluster_centers_[:,0])):
            curr_feature_z = []
            for y in range(len(thalamus_kmeans.cluster_centers_[0,:])):
                curr_z = (thalamus_kmeans.cluster_centers_[x,y] -
                          np.mean(thalamus_kmeans.cluster_centers_[:,y])) \
                          / np.std(thalamus_kmeans.cluster_centers_[:,y])
                curr_feature_z.append(curr_z)
            if x == 0:
                all_features_z = curr_feature_z
            else:
                all_features_z = np.vstack((all_features_z, curr_feature_z))
        if hemi == '0':
            hemi_name = 'lh'
        elif hemi == '1':
            hemi_name = 'rh'
        fname = '{0}_all_features_{1}-clus.csv'.format(hemi_name, n_clus)
        np.savetxt(fname, all_features_z)
        out_features.append(os.path.abspath(fname))

        curr_hemi_limbicthal_targ_clusters = []

        for i, curr_feat in enumerate(all_features_z):

            #LABELS (see below)
            #MTL: EC=1, pHC=9, Amyg = 16, HC=17, Nucleus Accumbens=18
            #mPFC:  mOrb=5, rACC = 13
            #controls: Paracentral=8, postcentral= 11, precentral=12, [for later script parietal=10, Superfront = 14]

            if ((curr_feat[1] > 0.0 and curr_feat[9] > 0.0 and curr_feat[16] > 0.0 and curr_feat[17] > 0.0 and curr_feat[18]) and (curr_feat[5] > 0.0 and curr_feat[13] > 0.0)) and not (curr_feat[8] > 0.0 or curr_feat[11] > 0.0 or curr_feat[12] > 0.0 or curr_feat[10] > 0.0 or curr_feat[14] > 0.0):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)
            elif ((curr_feat[1] > 0.0 or curr_feat[9] > 0.0 or curr_feat[16] > 0.0 or curr_feat[17] > 0.0 or curr_feat[18]) and (curr_feat[5] > 0.0 or curr_feat[13] > 0.0)) and not (curr_feat[8] > 0.0 or curr_feat[11] > 0.0 or curr_feat[12] > 0.0 or curr_feat[10] > 0.0 or curr_feat[14] > 0.0):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)
            elif ((curr_feat[1] > 0.0 or curr_feat[9] > 0.0 or curr_feat[16] > 0.0 or curr_feat[17] > 0.0 or curr_feat[18]) or (curr_feat[5] > 0.0 or curr_feat[13] > 0.0)) and not (curr_feat[8] > 0.0 or curr_feat[11] > 0.0 or curr_feat[12] > 0.0 or curr_feat[10] > 0.0 or curr_feat[14] > 0.0):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)


#Check the order of your target regions. In our data it was saved as:
    #curr_feat#		brain regions
    #curr_feat[0]	caudantcing
    #curr_feat[1]	EC
    #curr_feat[2]	fusi
    #curr_feat[3]	latfront
    #curr_feat[4]	medoccip
    #curr_feat[5]	medorbfront
    #curr_feat[6]	medpost
    #curr_feat[7]	occip
    #curr_feat[8]	paracentral
    #curr_feat[9]	paraHC
    #curr_feat[10]	parietal
    #curr_feat[11]	postcentral
    #curr_feat[12]	precentral
    #curr_feat[13]	rostralACC
    #curr_feat[14]	superFront
    #curr_feat[15]	temporal
    #curr_feat[16]	amygdala
    #curr_feat[17]	HC
    #curr_feat[18]	NAcc


        # Deal with when no clusters fit the restricted definition above
        # put in a dummy target_cluster = 999
        if len(curr_hemi_limbicthal_targ_clusters) == 0:
            curr_hemi_limbicthal_targ_clusters.append(999)

        target_clusters.append(curr_hemi_limbicthal_targ_clusters)
    return out_files, target_clusters, out_features

def extract_lh_values(target_clusters):
    match_values = target_clusters[0]
    for i, item in enumerate(match_values):
        if len(item) == 0:
            match_values[i] = [999]
    return match_values

def extract_rh_values(target_clusters):
    match_values = target_clusters[1]
    for i, item in enumerate(match_values):
        if len(item) == 0:
            match_values[i] = [999]
    return match_values

def select_lh_file(out_files):
    lh_file = out_files[0]
    return lh_file

def select_rh_file(out_files):
    rh_file = out_files[1]
    return rh_file

def pickfirst(in_file):
    if isinstance(in_file, list):
        return in_file[0]
    else:
        return in_file


sink_directory = '/home/data/madlab/data/mri/hcp/kmeans/kmeans_final_masks/hcp_kmeans_mthal_072821_5controls'
work_directory = '/scratch/madlab/puck/hcp_kmeans_5controls/work_dir'
data_dir = '/home/data/madlab/data/mri/hcp/probtrack/thalamus_25ksamp_avoidmask_january'

sids = os.listdir(data_dir)
#sids = ['100307']

hcp_thal_wf = pe.Workflow(name='hcp_thal_wf')
hcp_thal_wf.base_dir = work_directory


info = dict(fs_brain=[['subject_id', 'T1w_acpc_dc']],
            mni_brain=[['subject_id', 'T1w_restore_brain']],
            dmri_brain=[['subject_id','T1w_acpc_dc_restore_1.25']],
            aparcaseg=[['subject_id','aparc+aseg']],
            acpc2standard_warp=[['subject_id', 'acpc_dc2standard']])

subj_iterable = pe.Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True), name='subj_iterable')
subj_iterable.iterables = ('subject_id', sids)

# create a datasource doe to get the standard space brain and warp to standard space
datasource = pe.Node(nio.DataGrabber(infields=['subject_id'],outfields=list(info.keys())),
                     name='datasource')
datasource.inputs.base_directory = os.path.abspath('/home/data/hcp')
datasource.inputs.field_template = dict(fs_brain='%s/T1w/%s.nii.gz',
                                        mni_brain='%s/MNINonLinear/%s.nii.gz',
                                        dmri_brain='%s/T1w/%s.nii.gz',
                                        aparcaseg='%s/T1w/%s.nii.gz',
                                        acpc2standard_warp='%s/MNINonLinear/xfms/%s.nii.gz')
datasource.inputs.template = '%s/%s'
datasource.inputs.sort_filelist = True
datasource.inputs.template_args = info
hcp_thal_wf.connect(subj_iterable, 'subject_id', datasource, 'subject_id')

# Create a flirt node to calculate the dmri_brain to fs_brain xfm
dmri2fs_xfm = pe.Node(fsl.FLIRT(), name = 'dmri2fs_xfm')
dmri2fs_xfm.inputs.out_matrix_file = 'dmri_2_fs_xfm.mat'
hcp_thal_wf.connect(datasource, 'dmri_brain', dmri2fs_xfm, 'in_file')
hcp_thal_wf.connect(datasource, 'fs_brain', dmri2fs_xfm, 'reference')

# Create a convertxfm node to create inverse xfm of dmri2fs affine
invt_dmri2fs = pe.Node(fsl.ConvertXFM(), name='invt_dmri2fs')
invt_dmri2fs.inputs.invert_xfm = True
invt_dmri2fs.inputs.out_file = 'fs_2_dmri_xfm.mat'
hcp_thal_wf.connect(dmri2fs_xfm, 'out_matrix_file', invt_dmri2fs, 'in_file')

# Extract thalamus seed masks from aparc+aseg.nii.gz file
thal_seed_mask = pe.MapNode(fs.Binarize(), iterfield=['match', 'binary_file'], name = 'thal_seed_mask')
thal_seed_mask.inputs.match = [[10],[49]]
thal_seed_mask.inputs.binary_file = ['lft_thal.nii.gz', 'rt_thal.nii.gz']
hcp_thal_wf.connect(datasource, 'aparcaseg', thal_seed_mask, 'in_file')

# Create a flirt node to apply inverse transform to seeds
thalmaskxfm_fs2dmri = pe.MapNode(fsl.FLIRT(),
                                 iterfield = ['in_file'],
                                 name='thalmaskxfm_fs2dmri')
thalmaskxfm_fs2dmri.inputs.apply_xfm = True
thalmaskxfm_fs2dmri.inputs.interp = 'nearestneighbour'
hcp_thal_wf.connect(thal_seed_mask, 'binary_file', thalmaskxfm_fs2dmri, 'in_file')
hcp_thal_wf.connect(datasource, 'dmri_brain', thalmaskxfm_fs2dmri, 'reference')
hcp_thal_wf.connect(invt_dmri2fs, 'out_file', thalmaskxfm_fs2dmri, 'in_matrix_file')

# create a function node that runs the kmeans algorithm
run_kmeans = pe.Node(util.Function(input_names=['subject_id', 'n_clus', 'thal_masks'],
                                   output_names=['out_files', 'target_clusters', 'out_features'],
                                   function=kmeans_func),
                        name='run_kmeans')
run_kmeans.inputs.n_clus = n_clus
hcp_thal_wf.connect(subj_iterable, 'subject_id', run_kmeans, 'subject_id')
hcp_thal_wf.connect(thalmaskxfm_fs2dmri, 'out_file', run_kmeans, 'thal_masks')

# create a node to binarize LH the targeted clusters (aka midline clusters)
midline_thal_bin = pe.MapNode(fs.Binarize(),
                              iterfield=['in_file', 'match', 'binary_file'],
                              name='midline_thal_bin')
midline_thal_bin.inputs.binary_file = ['lft_limbic_thal_bin.nii.gz', 'rt_limbic_thal_bin.nii.gz']
hcp_thal_wf.connect(run_kmeans, 'out_files', midline_thal_bin, 'in_file')
hcp_thal_wf.connect(run_kmeans, 'target_clusters', midline_thal_bin, 'match')

# Add together the left and right hemisphere masks for a single mask
bi_limbic_thal_mask_combine = pe.Node(fsl.ImageMaths(op_string='-add'),
                                      name='bi_limbic_thal_mask_combine')
bi_limbic_thal_mask_combine.inputs.out_file = 'bi_limbic_thal_bin.nii.gz'
hcp_thal_wf.connect(midline_thal_bin, ('binary_file',select_lh_file), bi_limbic_thal_mask_combine, 'in_file')
hcp_thal_wf.connect(midline_thal_bin, ('binary_file',select_rh_file), bi_limbic_thal_mask_combine, 'in_file2')

# Create an applywarp node to warp from acpc to mni standard space
limbic_thal_bin_acpc2mni_warp = pe.MapNode(fsl.ApplyWarp(),
                                           iterfield=['in_file'],
                                           name='limbic_thal_bin_acpc2mni_warp')
limbic_thal_bin_acpc2mni_warp.inputs.interp = 'nn'
hcp_thal_wf.connect(midline_thal_bin, 'binary_file', limbic_thal_bin_acpc2mni_warp, 'in_file')
hcp_thal_wf.connect(datasource, 'mni_brain', limbic_thal_bin_acpc2mni_warp, 'ref_file')
hcp_thal_wf.connect(datasource, 'acpc2standard_warp', limbic_thal_bin_acpc2mni_warp, 'field_file')

# Create an applywarp node to warp from acpc to mni standard space
bi_limbic_thal_mask_acpc2mni_warp = pe.Node(fsl.ApplyWarp(),
                                            name='bi_limbic_thal_mask_acpc2mni_warp')
bi_limbic_thal_mask_acpc2mni_warp.inputs.interp = 'nn'
hcp_thal_wf.connect(bi_limbic_thal_mask_combine, 'out_file', bi_limbic_thal_mask_acpc2mni_warp, 'in_file')
hcp_thal_wf.connect(datasource, 'mni_brain', bi_limbic_thal_mask_acpc2mni_warp, 'ref_file')
hcp_thal_wf.connect(datasource, 'acpc2standard_warp', bi_limbic_thal_mask_acpc2mni_warp, 'field_file')

# create a node to merge the binarized files in standard space
lh_thal_merge = pe.JoinNode(fsl.Merge(),
                            joinsource='subj_iterable',
                            joinfield='in_files',
                            name='lh_thal_merge')
lh_thal_merge.inputs.dimension = 't'
lh_thal_merge.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(limbic_thal_bin_acpc2mni_warp, ('out_file',select_lh_file), lh_thal_merge, 'in_files')

# create a node to merge the binarized files in standard space
rh_thal_merge = pe.JoinNode(fsl.Merge(),
                            joinsource='subj_iterable',
                            joinfield='in_files',
                            name='rh_thal_merge')
rh_thal_merge.inputs.dimension = 't'
rh_thal_merge.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(limbic_thal_bin_acpc2mni_warp, ('out_file',select_rh_file), rh_thal_merge, 'in_files')

# create a node to merge the binarized files in standard space
bi_thal_merge = pe.JoinNode(fsl.Merge(), joinsource='subj_iterable',
                            joinfield='in_files', name='bi_thal_merge')
bi_thal_merge.inputs.dimension = 't'
bi_thal_merge.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(bi_limbic_thal_mask_acpc2mni_warp, 'out_file', bi_thal_merge, 'in_files')

# create a node to calculate the mean image for LH thal mask
lh_thal_mean = pe.Node(fsl.MeanImage(), name='lh_thal_mean')
lh_thal_mean.inputs.dimension = 'T'
lh_thal_mean.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(lh_thal_merge, 'merged_file', lh_thal_mean, 'in_file')

# create a node to calculate the mean image for RH thal mask
rh_thal_mean = pe.Node(fsl.MeanImage(), name='rh_thal_mean')
rh_thal_mean.inputs.dimension = 'T'
rh_thal_mean.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(rh_thal_merge, 'merged_file', rh_thal_mean, 'in_file')

# create a node to calculate the mean image for COMBINED thal mask
bi_thal_mean = pe.Node(fsl.MeanImage(), name='bi_thal_mean')
bi_thal_mean.inputs.dimension = 'T'
bi_thal_mean.inputs.output_type = 'NIFTI_GZ'
hcp_thal_wf.connect(bi_thal_merge, 'merged_file', bi_thal_mean, 'in_file')

# create a datasink node to save everything
datasink = pe.Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = os.path.abspath(sink_directory)
datasink.inputs.substitutions = [('_subject_id_', '')]
hcp_thal_wf.connect(subj_iterable, 'subject_id', datasink, 'container')
hcp_thal_wf.connect(datasource, 'mni_brain', datasink, 'anat.@mni_brain')
hcp_thal_wf.connect(datasource, 'dmri_brain', datasink, 'anat.@dmri_brain')
hcp_thal_wf.connect(midline_thal_bin, 'binary_file', datasink, 'dmri_space.@limbic_thal')
hcp_thal_wf.connect(bi_limbic_thal_mask_combine, 'out_file', datasink, 'dmri_space.@bilimbic_thal')

hcp_thal_wf.connect(limbic_thal_bin_acpc2mni_warp, 'out_file', datasink, 'mni_space.@limbic_thal')
hcp_thal_wf.connect(bi_limbic_thal_mask_acpc2mni_warp, 'out_file', datasink, 'mni_space.@bilimbic_thal')
hcp_thal_wf.connect(lh_thal_mean, 'out_file', datasink, 'avgmasks.@lhmask')
hcp_thal_wf.connect(rh_thal_mean, 'out_file', datasink, 'avgmask.@rhmask')
hcp_thal_wf.connect(bi_thal_mean, 'out_file', datasink, 'avgmask.@bimask')
hcp_thal_wf.connect(run_kmeans, 'out_features', datasink, 'kmeans.@features')
hcp_thal_wf.connect(run_kmeans, 'out_files', datasink, 'kmeans.@masks')

hcp_thal_wf.config['execution']['crashdump_dir'] = '/scratch/madlab/puck/hcp_kmeans_5controls/crash'
hcp_thal_wf.config['execution']['crashfile_format'] = 'txt'
hcp_thal_wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p IB_44C_512G --qos pq_madlab --account iacc_madlab -N 1 -n 1'), 'overwrite': True})
