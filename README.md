# Robotool Data Collection

move the file in /scripts_avp to /avp_human_data

create an conda env robotool or install needed packages in your environment 

## Collection Process

1. run  roscore in background
2. change the saving path in line 178-181 of collect_whole_data.py

and run in avp repo:
'''
    python collect_whole_data.py --exp_name <TASK_NAME>
'''

3. change the saving path in line 476 of test_cameras_avp_optimized.py

and run in robotool repo:
'''
    python test_cameras_avp_optimized.py --name <TASK_NAME>
'''
* the avp should be run first!

## Data Processing

after collecting data, run convert_npy_to_h5.py to generate the needed data format.

## Update on exp management

now run managed_collect_whole_data and managed_collect_robotool_data with --task_name and --exp_name for file management.

the saved file will be in same directory and managed automatically.