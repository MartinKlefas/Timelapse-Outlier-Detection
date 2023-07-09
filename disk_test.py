import subprocess,re,pathlib

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


# Check if there are any error messages
if disk_drive_process.stderr:
    print(f"Disk Drive Error: {disk_drive_process.stderr}")

else:
    # parse disk drive informaiton
    lines = disk_drive_process.stdout.strip().split('\n')
    
    header = [col.strip() for col in re.split(r" {2,}",lines[0])]
    
    disk_drive_info = {}
    for line in lines[1:]:
        if len(line.strip()) >0 :
            values = [value.strip() for value in re.split(r" {2,}",line)]
            values[0] = str(values[0]).replace("\\", "").replace(".","")
            disk_drive_info[values[0]] = dict(zip(header[1:], values[1:]))


drive_mappings = get_drive_mappings()
drive_letter_info = {}
# Print the disk drive to drive letter mappings
for drive_letter, disk_drive in drive_mappings.items():

    trimmed_drive = re.search(r"PHYSICALDRIVE\d+", disk_drive).group(0)
    
    drive_letter_info[drive_letter] = disk_drive_info[trimmed_drive]

def get_drive_info(drive : str = "", Path : pathlib.Path = None):
    if Path :
        Path = Path.resolve()
        drive = str(Path.drive)

    drive = drive.replace(":","")
    return drive_letter_info[drive]