import os
import csv

from .const import *


def read_dataset_text():
    dataset = {}

    for folder in sub_folders_text:
        file_path = os.path.join(folder, text_file_name)

        if not os.path.exists(file_path):
            continue

        folder_name = os.path.basename(os.path.dirname(file_path))

        # Some files are encoded with ISO 8859-1, some are UTF-8.
        # Trying to work around this dataset by first trying UTF-8,
        # then ISO 8859-1 won't work because some files are valid UTF-8
        # even though they were encoded with ISO 8859-1.
        # To work around that, just hardcode which files contain UTF-8.
        # 0099 is just completely broken.
        if folder_name in ("0099"):
            continue
        if folder_name in ("0090", "0101", "0102"):
            encoding = 'utf-8'
        else:
            encoding = 'iso-8859-1'

        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        parsed_file = []
        for line in lines:
            # print('>', line, '<', file_path)
            start_time, end_time, sentence, number = line.strip().split(";")
            assert number == "1", f"number is {number}"

            parsed_file.append((start_time, end_time, sentence, number))
        #print(parsed_file)
        dataset[folder_name] = parsed_file

    return dataset


def read_dataset_mms():
    dataset = {}

    for file_name in sorted(os.listdir(main_folder_mms)):
        file_path = os.path.join(main_folder_mms, file_name)
        file_number = file_name.rsplit('.', maxsplit=1)[0]

        parsed_file = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            #print(f'reading file {file_path}')
            reader = csv.DictReader(f)
            for row in reader:
                parsed_file.append(row)

        dataset[file_number] = parsed_file

    return dataset


def write_dataset_text(dataset, main_folder):
    for folder_name, file_contents in dataset.items():
        file_folder = os.path.join(main_folder, folder_name)
        os.makedirs(file_folder, exist_ok = True)
        file_path = os.path.join(file_folder, text_file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            for row in file_contents:
                text_line = ";".join(row)
                f.write(text_line + '\n')


def write_dataset_mms(dataset, main_folder):
    fieldnames = ['maingloss', 'framestart', 'frameend', 'duration', 'transition', 'domgloss', 'ndomgloss', 'domreloc', 'ndomreloc', 'headpos', 'headmov', 'cheecks', 'nose', 'mouthgest', 'mouthing', 'eyegaze', 'eyeaperture', 'eyebrows', 'neck', 'shoulders', 'torso', 'domhandrelocx', 'domhandrelocy', 'domhandrelocz', 'domhandrelocax', 'domhandrelocay', 'domhandrelocaz', 'domhandrelocsx', 'domhandrelocsy', 'domhandrelocsz', 'domhandrotx', 'domhandroty', 'domhandrotz', 'ndomhandrelocx', 'ndomhandrelocy', 'ndomhandrelocz', 'ndomhandrelocax', 'ndomhandrelocay', 'ndomhandrelocaz', 'ndomhandrelocsx', 'ndomhandrelocsy', 'ndomhandrelocsz', 'ndomhandrotx', 'ndomhandroty', 'ndomhandrotz']
    os.makedirs(main_folder, exist_ok = True)

    for file_name, file_contents in dataset.items():
        file_path = os.path.join(main_folder, file_name + '.mms')
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in file_contents:
                writer.writerow(row)
