import os

def delete_files_in_directory(directory):
    file_list = os.listdir(directory)
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

delete_files_in_directory("./testing")
delete_files_in_directory("./validation")
delete_files_in_directory("./training")