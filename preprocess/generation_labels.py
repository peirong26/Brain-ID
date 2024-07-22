"""
Data preprocessing for Brain-ID training

Steps
1) SynthSR: map to a T1 contrast for better Skull-stripping performance;
2) SynthStrip: skull-stripping; saving the stripped T1 as the supervision ground truth;
3) SynthSeg: get segmentation labels;
4) Merge left & right labels for obtaining the anatomy generation labels.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.misc as utils


# NOTE: activate your freesurfer env first
os.system('export FREESURFER_HOME=/usr/local/freesurfer/7.4.1/')
os.system('source $FREESURFER_HOME/SetUpFreeSurfer.sh')



# your downloaded images
root_dir = '/path/to/your/dataset'
img_dir = os.path.join(root_dir, 'T1_orig') 

# pre-processing
img_strip_save_dir = utils.make_dir(os.path.join(root_dir, 'T1')) 
synthsr_save_dir = utils.make_dir(os.path.join(root_dir, 'synthsr'))
synthstrip_save_dir = utils.make_dir(os.path.join(root_dir, 'synthstrip'))
mask_save_dir = utils.make_dir(os.path.join(root_dir, 'brainmask'))

# obtaining the anatomy labels 
synthseg_save_dir = utils.make_dir(os.path.join(root_dir, 'label_maps_segmentation'))
gen_save_dir = utils.make_dir(os.path.join(root_dir, 'label_maps_generation'))



subjs = os.listdir(img_dir)
subjs.sort()

for i, name in enumerate(subjs): 
    if name.startswith('sub-') and (name.endswith('.nii') or name.endswith('.nii.gz')):
        basename = name.split('.')[0]
        print('Now processing: %s (%d/%d)' % (name, i+1, len(subjs)))

        if not os.path.isfile(os.path.join(synthsr_save_dir, basename + '.nii')):
            print('  synthsr-ing')
            os.system('mri_synthsr' + ' --i ' + os.path.join(img_dir, name) + ' --o ' + os.path.join(synthsr_save_dir, basename + '.nii'))

        if not os.path.isfile(os.path.join(img_strip_save_dir, basename + '.nii')):
            print('  synthstrip-ing')
            os.system('mri_synthstrip' + ' -i ' + os.path.join(synthsr_save_dir, basename + '.nii') + ' -o ' + os.path.join(synthstrip_save_dir, basename + '.nii') + ' -m ' + os.path.join(mask_save_dir, basename + '.nii'))
            mask, aff = utils.MRIread(os.path.join(mask_save_dir, basename + '.nii'))
            img, aff = utils.MRIread(os.path.join(img_dir, name))
            img *= mask # skull-stripping
            utils.MRIwrite(img, aff, os.path.join(img_strip_save_dir, basename + '.nii'))

        if not os.path.isfile(os.path.join(synthseg_save_dir, basename + '.nii')):
            print('  synthseg-ing & resampling')
            os.system('mri_synthseg' + ' --i ' + os.path.join(img_strip_save_dir, basename + '.nii') + ' --o ' + os.path.join(synthseg_save_dir, basename + '.nii' + ' --resample ' + os.path.join(img_strip_save_dir, basename + '.nii')))

        if not os.path.isfile(os.path.join(gen_save_dir, basename + '.nii')):
            print('  genseg-ing')
            im, aff = utils.MRIread(os.path.join(synthseg_save_dir, basename + '.nii'), im_only=False, dtype='float')
            for l in utils.right_to_left_dict.keys():
                im[im==l] = utils.right_to_left_dict[l]
            utils.MRIwrite(im, aff, os.path.join(gen_save_dir, basename + '.nii'))
