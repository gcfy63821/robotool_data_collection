#!/usr/bin/env bash

# need to copy

#local_base_dir="/home/weiyu/data_drive/kdm/calvin/dataset/"
#remote_base_dir="/svl/u/weiyul/data_drive/kdm/calvin/dataset/"
#folder_name="task_D_D"
#zip_file=${folder_name}.tar.gz
#
#cd $local_base_dir
## tar -czf "${folder_name}".tar.gz $folder_name
#tar cf - $folder_name -P | pv -s $(du -sb "${folder_name}" | awk '{print $1}') | gzip > $zip_file
#rsync -aP $zip_file weiyul@scdt:$remote_base_dir
#
#echo "cd ${remote_base_dir} && tar -xzf ${zip_file}"
#echo "rm ${zip_file}"

local_base_dir="/home/weiyu/data_drive/kdm/real_world/"
remote_base_dir="/viscam/projects/kdm/real_world_processed/"
folder_name="train_new"
zip_file=${folder_name}.tar.gz

cd $local_base_dir
# tar -czf "${folder_name}".tar.gz $folder_name
tar cf - $folder_name -P | pv -s $(du -sb "${folder_name}" | awk '{print $1}') | gzip > $zip_file
rsync -aP $zip_file weiyul@scdt:$remote_base_dir

echo "cd ${remote_base_dir} && tar -xzf ${zip_file}"
echo "rm ${zip_file}"

