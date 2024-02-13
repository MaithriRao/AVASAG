# Run with: python -m augmented_data.location

import nltk
import spacy
import random
from .utils import replace_multiple, normalize_text_to_mms


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

nlp = spacy.load('de_core_news_sm')


def replace_location_entities(dataset_text, dataset_mms):
    location_names = set()
    dataset_text_with_metadata = {}

    excludes = {'Zuges', 'Alternativen', 'D', 'RE 77','A.', 'D.','Umsteigen','Weiteres',
            'Notbremse', 'Reservierungen','IC 2313','Sonderzug','Rhein'}


    for folder_name, file in dataset_text.items():
        file_with_metadata = []
        for (start_time, end_time, sentence, number) in file:
            sentences_to_analyze = sentence.strip().translate({ord(i): None for i in "„“"})
            # sentences_to_analyze = sentence.strip().replace("„", "").replace("“", "")
            doc = nlp(sentences_to_analyze)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            # print(entities)
            # print(folder_name, sentence)

            for ent in doc.ents:
                if ent.label_ == 'LOC' and ent.text not in excludes:
                    # Check if "Hauptbahnhof" is present in the entity text
                    location = ent.text
                    if 'Hauptbahnhof' in location:
                        location = location.replace('Hauptbahnhof', '').strip()  # Remove "Hauptbahnhof" and strip extra spaces
                    location_names.add(location)
                    # print('location_names are', location_names)
            file_with_metadata.append((start_time, end_time, sentence, number, entities))

        dataset_text_with_metadata[folder_name] = file_with_metadata
        # print("file_with_metadata", dataset_text_with_metadata)

    # print(location_names)  # Finding all the locations

    result_text = {}
    result_mms = {}

    for folder_name, file in dataset_text_with_metadata.items():
        location_counts = {}
        for line_number, (start_time, end_time, sentence, number, entities) in enumerate(file):
            for (text, label) in entities:
                if label == 'LOC' and text not in excludes:
                    if 'Hauptbahnhof' in text:
                        text = text.replace('Hauptbahnhof', '').strip()
                    location_counts[text] = location_counts.get(text, 0) + 1 #counting the number of times same location appears in a file
                    # print(f'WARNING: location {text} in file {folder_name} appears multiple times')

        mapping = {}
        for location, count in location_counts.items():
            assert len(location_names) > 1, f'ERROR: only one location found'
            while True:
                new_location = random.choice(tuple(location_names))
                if new_location != location:
                    break
            mapping[location] = new_location


        new_text_data = []
        for (start_time, end_time, sentence, number, entities) in file:
            sentence, _ = replace_multiple(sentence, mapping)
            new_text_data.append((start_time, end_time, sentence, number))
        result_text[folder_name] = new_text_data

        replaced_counts = {}
        new_mms_data = []
        for row in dataset_mms[folder_name]:
            mapping_mms = dict((normalize_text_to_mms(k), normalize_text_to_mms(v)) for k, v in mapping.items())
            new_row = row.copy()
            word = row['maingloss']
            if word in mapping_mms:
                new_row['maingloss'] = mapping_mms[word]
                replaced_counts[word] = replaced_counts.get(word, 0) + 1
            new_mms_data.append(new_row)
        result_mms[folder_name] = new_mms_data

        for location, count in location_counts.items():
            location_mms = normalize_text_to_mms(location)
            replaced_count = replaced_counts.get(location_mms, 0)
            if replaced_count != count:
                print(f'WARNING: replaced_count in file {folder_name} should be {count} but was {replaced_count}, trying to replace {location_mms}')

    return (result_text, result_mms)



if __name__ == "__main__":
    from .dataset import *
    dataset_text = read_dataset_text()
    dataset_mms = read_dataset_mms()

    result_text, result_mms = replace_location_entities(dataset_text, dataset_mms)
    write_dataset_text(result_text, main_folder = 'modified/location/text')
    write_dataset_mms(result_mms, main_folder = 'modified/location/mms')
