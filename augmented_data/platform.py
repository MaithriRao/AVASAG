# Run with: python -m augmented_data.platform

import re
import random
from enum import Enum, verify, UNIQUE
from collections import Counter
from .utils import replace_multiple, parse_num


def replace_platform_entities(dataset_text, dataset_mms):
    all_platforms = set()
    dataset_text_with_metadata = {}

    platform_pattern = r'\b(Gleis \d+)[a-z]?\b'

    for folder_name, file in dataset_text.items():
        platforms_per_file = []
        for (start_time, end_time, sentence, number) in file:
            platforms_per_file += re.findall(platform_pattern, sentence)

        all_platforms = all_platforms.union(set(platforms_per_file))
        platform_counts = Counter(platforms_per_file)
        dataset_text_with_metadata[folder_name] = (file, platform_counts)

    result_text = {}
    result_mms = {}
    for folder_name, file_with_metadata in dataset_text_with_metadata.items():
        mapping = {}
        (file, platform_counts) = file_with_metadata
        all_platforms_tuple = tuple(all_platforms)
        for platform, count in platform_counts.items(): # TODO: think about randomly shuffling instead
            assert len(all_platforms) > 1, f'ERROR: only one platform found'
            while True:
                new_platform = random.choice(all_platforms_tuple)
                if new_platform != platform:
                    break
            mapping[platform] = new_platform

        new_text_data = []
        for start_time, end_time, sentence, number in file:
            sentence, _ = replace_multiple(sentence, mapping)
            new_text_data.append((start_time, end_time, sentence, number))
        result_text[folder_name] = new_text_data

        @verify(UNIQUE)
        class State(Enum):
            NOT_FOUND = 0
            GLEIS = 1
            WECHSELN = 2
            NUM = 3

        replaced_counts = {}
        new_mms_data = []
        state = State.NOT_FOUND
        for row in dataset_mms[folder_name]:
            new_row = row.copy()
            word = row['maingloss']
            if state == State.NOT_FOUND:
                if word == 'GLEIS':
                    state = State.GLEIS
            elif state == State.GLEIS or state == State.WECHSELN:
                if state == State.GLEIS and word == 'WECHSELN':
                    state = State.WECHSELN
                elif word.startswith('num:'):
                    state = State.NUM
                else:
                    print(f'WARNING: expected WECHSELN or num:, got {word} in file {folder_name}')
                    state = State.NOT_FOUND

            if state == State.NUM:
                num = parse_num(word, folder_name)
                old_gleis = f'Gleis {num}'
                new_gleis = mapping[old_gleis]
                print(f'Found {old_gleis} in file {folder_name}, replacing with {new_gleis}')
                new_num = new_gleis.removeprefix('Gleis ')
                new_row['maingloss'] = 'num:' + new_num
                replaced_counts[old_gleis] = replaced_counts.get(old_gleis, 0) + 1
                state = State.NOT_FOUND

            new_mms_data.append(new_row)
        result_mms[folder_name] = new_mms_data

        for platform, count in platform_counts.items():
            replaced_count = replaced_counts.get(platform, 0)
            if replaced_count != count:
                print(f'WARNING: replaced_count in file {folder_name} should be {count} but was {replaced_count}, trying to replace {platform}')

    # print(result_text)
    # print(all_platforms)
    return (result_text, result_mms)



if __name__ == "__main__":
    from .dataset import *
    dataset_text = read_dataset_text()
    dataset_mms = read_dataset_mms()

    result_text, result_mms = replace_platform_entities(dataset_text, dataset_mms)
    write_dataset_text(result_text, main_folder = 'modified/platform/text')
    write_dataset_mms(result_mms, main_folder = 'modified/platform/mms')
