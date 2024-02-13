import os


main_folder_text = "annotations_full/annotations"
main_folder_mms = "mms-subset91"

sub_folders_text = sorted([f.path for f in os.scandir(main_folder_text) if f.is_dir()])

text_file_name = "gebaerdler.Text_Deutsch.annotation~"
