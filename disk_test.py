import subprocess

import win32com.client

def get_drive_mappings():
    wmi = win32com.client.GetObject("winmgmts:")
    mappings = {}

    for partition_to_disk_drive in wmi.InstancesOf("Win32_DiskDriveToDiskPartition"):
        disk_drive_path = partition_to_disk_drive.Antecedent
        partition_path = partition_to_disk_drive.Dependent

        for logical_disk_to_partition in wmi.InstancesOf("Win32_LogicalDiskToPartition"):
            if logical_disk_to_partition.Antecedent == partition_path:
                drive_letter = logical_disk_to_partition.Dependent[-3]
                mappings[drive_letter] = disk_drive_path

    return mappings

# Execute the 'wmic' command and store the output for disk drives
disk_drive_process = subprocess.run('wmic diskdrive get DeviceID, Model, MediaType, InterfaceType',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

# Execute the 'wmic' command and store the output for logical disks
#logical_disk_process = subprocess.run('wmic logicaldisk get DeviceID, DriveType, FileSystem, FreeSpace, Size, VolumeName',
#                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

# Check if there are any error messages
if disk_drive_process.stderr:
    print(f"Disk Drive Error: {disk_drive_process.stderr}")
#elif logical_disk_process.stderr:
#    print(f"Logical Disk Error: {logical_disk_process.stderr}")
else:
    # Print the output for disk drives
    print("Disk Drives:")
    print(disk_drive_process.stdout)

    # Print the output for logical disks
#    print("Logical Disks:")
#    print(logical_disk_process.stdout)




drive_mappings = get_drive_mappings()

# Print the disk drive to drive letter mappings
for drive_letter, disk_drive in drive_mappings.items():
    print(f"{drive_letter} -> {disk_drive}")
