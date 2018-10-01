import quantities as pq

'''
This is an example parameter file to use with expipe-plugin-cinpla.

Reserved names:
    subject
    electrophysiology
    implantation
    injection
'''


#  these are templates you want to add to each recording perfomed with axona
axona_templates = [
    'hardware_axona_tracker',
    'hardware_axona_daq',
    'hardware_axona_camera_objective',
    'environment_open_field_tracking',
    'environment_rat_housing',
    'hardware_optic_fibre',
    'hardware_tetrode',
    'hardware_microdrive_optetrode'
]


#  these are templates you want to add to each recording perfomed with
#  open ephys
openephys_templates = [
    'hardware_openephys_daq',
    'hardware_intan_headstage',
    'hardware_pointgrey_camera_objective',
    'software_bonsai_gui',
    'software_openephys_gui',
    'environment_open_field_tracking',
    'environment_rat_housing',
    'hardware_optic_fibre',
    'hardware_tetrode',
    'hardware_microdrive_optetrode'
]


#  these are templates you want to add to each optogenetics session perfomed
#  with openephys
opto_openephys_templates = [
    'mikkel_laser_settings',
    'mikkel_pulse_pal_settings',
    'mikkel_optogenetics_paradigm',
    'mikkel_optogenetics_anatomical_location'
    'hardware_openephys_optogenetics',
    'hardware_pulse_pal',
    'hardware_blue_laser',
    'hardware_laser_measure_device',
]

#  these are templates you want to add to each optogenetics session perfomed
#  with axona
opto_axona_templates = [
    'mikkel_laser_settings',
    'mikkel_pulse_pal_settings',
    'mikkel_optogenetics_paradigm',
    'mikkel_optogenetics_anatomical_location',
    'hardware_axona_optogenetics',
    'hardware_pulse_pal',
    'hardware_blue_laser',
    'hardware_laser_measure_device',
]

#  thse are e.g. the brain areas you investigate, this ensures that you
#  register with with the same nomenclature for every recording
POSSIBLE_BRAIN_AREAS = ['MECR', 'MECL', 'MS']
POSSIBLE_OPTO_TAGS = ['opto-inside', 'opto-outside', 'opto-train']
POSSIBLE_LOCATIONS = ['room2', 'room1']
POSSIBLE_CELL_LINES = ['gad1-cre', 'wild-type']

#  obligatory tags will be enforced uppon transfer, this is to ensure you
#  have a minimum of tags for each action
OBLIGATORY_TAGS = ['no', 'yes', 'maybe']
POSSIBLE_TAGS = OBLIGATORY_TAGS + [
    'GC', 'PC', 'BC', 'SC', 'HD', 'TC', 'theta', 'baseline',
    'AAV5-syn-hChR2-EYFP', 'AAV8-hsyn-eChR2(H134R)-EYFP',
    'AAV8-Ef1a-DIO-eChr2(H134R)-EYFP', 'baseline']

#  thse are the templates you want to load to each suregery implantation
#  procedure
surgery_implantation_templates = [
    'mikkel_anaesthesia',
    'mikkel_analgesia',
    'mikkel_analgesia_post',
    'mikkel_anaesthesia_local',
    'hardware_optic_fibre',
    'hardware_tetrode',
    'hardware_microdrive_optetrode',
    'environment_rat_housing',
    'environment_surgery_station'
]

# thse are the templates you want to load to each suregery injection procedure
surgery_injection_templates = [
    'mikkel_anaesthesia',
    # 'mikkel_anaesthesia_ketamine',
    'mikkel_analgesia',
    'mikkel_analgesia_post',
    'mikkel_anaesthesia_local',
    'environment_rat_housing',
    'environment_surgery_station'
]

perfusion_templates = [
    'mikkel_perfusion_procedure',
    'environment_surgery_station'
]

# this is how you represent each unit when you use expipe register-units
UNIT_INFO = {
    'info_waveform': {
        'alternatives': {
            'BS': 'definition: broad spiking waveform (putative excitatory)',
            'NS': 'definition: narrow spiking waveform (putative inhibitory)'
        },
        'value': ""
    },
    'info_pheno_type': {
        'value': ""
    }
}

# these are the analysis parameters you pass to exana
ANALYSIS_PARAMS = {
    'speed_filter': 5 * pq.m / pq.s,
    'pos_fs': 100 * pq.Hz,
    'f_cut': 6 * pq.Hz,
    'spat_binsize': 0.02 * pq.m,
    'spat_smoothing': 0.025,
    'grid_stepsize': 0.1 * pq.m,
    'box_xlen': 1 * pq.m,
    'box_ylen': 1 * pq.m,
    'ang_binsize': 4,
    'ang_n_avg_bin': 4,
    'imgformat': '.png',
    'corr_bin_width': 0.01 * pq.s,
    'corr_limit': 1. * pq.s,
    'isi_binsize': 1 * pq.ms,
    'isi_time_limit': 100 * pq.ms,
}

# this is personal user parameters
USER_PARAMS = {
    'project_id': 'mikkel_septum_entorhinal',
    'user_name': 'Mikkel Elle Lepperød',
    'location': 'room2',
    'laser_device': {'name': 'hardware_blue_laser', 'id': 4}
}

# this is a placeholder for all your templates
TEMPLATES = {
    'axona': axona_templates,
    'opto_axona': opto_axona_templates,
    'openephys': openephys_templates,
    'opto_openephys': opto_openephys_templates,
    'surgery_implantation': surgery_implantation_templates,
    'surgery_injection': surgery_injection_templates,
    'perfusion': perfusion_templates,
    'adjustment': ['mikkel_drive_depth_adjustment'],
    'implantation': {'mecl': 'mikkel_implant_drive_mecl',
                     'mecr': 'mikkel_implant_drive_mecr',
                     'ms': 'mikkel_implant_fibre_ms'},
    'injection': {'ms': 'mikkel_injection_ms'},
    'subject': 'mikkel_subject'
}
