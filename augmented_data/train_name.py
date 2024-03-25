# Run with: python -m augmented_data.train_name

import re
import random
from enum import Enum, verify, UNIQUE
from collections import defaultdict
from .utils import has_numbers, replace_multiple

# Exclude 0
single_sign_numbers = set()
for i in range(1, 20):
    single_sign_numbers.add(i)

for i in range(2, 10):
    single_sign_numbers.add(i * 10) # Tens
    single_sign_numbers.add(i * 11) # Repdigits

print(f'single_sign_numbers: {single_sign_numbers}')

class TrainData:
    def __init__(self):
        self.type_line = None
        self.train_type = None
        self.num_has_dashes = False
        self.num1_str = None
        self.num2_str = None
        self.num_int = None
        self.num1_line = None
        self.num2_line = None

    def is_valid(self):
        if self.train_type is None:
            print(f'ERROR: train_type has not been set yet, num: {self.num_int}')
            return False
        if self.num_int < 0: # TODO: think about setting an upper limit maybe?
            print(f'ERROR: not a valid number: {self.num_int}')
            self.num_int = None
            return False
        return True

    def set_train_type_mms(self, train_type, type_line):
        if train_type not in TrainType:
            raise Exception(f'ERROR: train_type is not a valid value: {train_type}')
        if self.train_type is not None:
            raise Exception('ERROR: train_type was already set')
        self.train_type = train_type
        self.type_line = type_line

    def set_train_type_text(self, train_type):
        if train_type not in TrainType:
            raise Exception(f'ERROR: train_type is not a valid value: {train_type}')
        if self.train_type is not None:
            raise Exception('ERROR: train_type was already set')
        self.train_type = train_type

    def set_num_mms(self, num_str, line_num):
        if self.num2_str is not None or self.num_has_dashes:
            raise Exception('ERROR: number was already set')
        if num_str == '':
            raise Exception('ERROR: number is empty')

        if '-' in num_str:
            num_str = num_str.replace('-', '')
            self.num_has_dashes = True
        try:
            num_int = int(num_str)
        except ValueError:
            raise Exception(f'ERROR: number is not a number: {num_str}')
        if num_int < 0:
            raise Exception(f'ERROR: number is negative: {num_int}')

        if self.num1_str is None:
            if not self.num_has_dashes and num_int not in single_sign_numbers:
                print(f'WARNING: num1 is not a valid single sign value: {num_str}')
            self.num1_str = num_str
            self.num1_line = line_num
        else:
            if num_str not in ['20', '30', '40', '50', '60', '70', '80', '90']:
                print(f'WARNING: num2 is not a valid value: {num_str}')
            if self.num_int not in range(1, 10):
                print(f'WARNING: num1 is not a valid value: {self.num1_str}')
            self.num2_str = num_str
            self.num2_line = line_num

        if self.num_int is None:
            self.num_int = 0
        self.num_int += num_int

    def set_num_text(self, num_str):
        if self.num1_str is not None:
            raise Exception('ERROR: number was already set')
        if num_str == '':
            raise Exception('ERROR: number is empty')
        self.num1_str = num_str
        try:
            num_int = int(num_str)
        except ValueError:
            raise Exception(f'ERROR: number is not a number: {num_str}')

        self.num_int = num_int

    def get_line_numbers(self):
        if self.type_line is None:
            raise Exception('ERROR: type has not been set yet')
        if self.num1_line is None:
            raise Exception('ERROR: number has not been set yet')
        return (self.type_line, self.num1_line, self.num2_line)

    def get_num_int(self):
        if self.num_int is None:
            raise Exception('ERROR: number has not been set yet')
        return self.num_int

    def get_train_type(self):
        if self.train_type is None:
            raise Exception(f'ERROR: train_type has not been set yet, num: {self.num_int}')
        return self.train_type


@verify(UNIQUE)
class TrainType(Enum):
    # Key is in the format of the text dataset
    # Value is in the format of the mms dataset
    UNKNOWN = "UNKNOWN"
    RE = "R-E"
    RB = "R-B"
    IC = "I-C"
    ICE = "ICE"

def pick_random_train_type(exclude=set()):
    exclude.add(TrainType.UNKNOWN)
    all_train_types = set(TrainType)
    filtered_train_types = all_train_types.difference(exclude)
    return random.choice(tuple(filtered_train_types))

def mms_train_num_to_str(num1_int, num2_int):
    num1_str = str(num1_int)
    if num1_int > 100: # Add dashes between the digits
        assert(num2_int is None)
        result = ''
        for digit in num1_str:
            if len(result) != 0:
                result += '-'
            result += digit
        return result
    return num1_str

def mms_train_type_to_str(train_type):
    assert train_type != TrainType.UNKNOWN
    if train_type == TrainType.ICE:
        return train_type.value
    return f'fa:{train_type.value}'

def assemble_train_text(train_type, num_int):
    return f'{train_type.name} {num_int}'


def replace_train_entities(dataset_text, dataset_mms):
    @verify(UNIQUE)
    class State(Enum):
        NOT_FOUND = 0
        TRAIN = 1
        FIRST_NUM = 2
        SECOND_NUM = 3
        TRAIN_FOUND = 4
        DONE_PARSING = 5

    def process_train(state, train_data, train_positions):
        if state == State.TRAIN_FOUND:
            state = State.DONE_PARSING
            if not train_data.is_valid():
                return (state, train_positions)
            # Here we know that the train number is valid
            (type_line, num1_line, num2_line) = train_data.get_line_numbers()
            num_int = train_data.get_num_int()
            train_type = train_data.get_train_type()
            print(f'Found train number: {train_type.value} {num_int}, lines: {num1_line}, {num2_line} (file_number: {file_number})')
            train_position = (type_line, train_type, num1_line, num2_line, num_int)
            train_positions_list = train_positions.get(file_number, [])
            train_positions_list.append(train_position)
            train_positions[file_number] = train_positions_list
        return (state, train_positions)

    result_mms = {}
    train_mappings = defaultdict(dict)
    train_counts = defaultdict(lambda: defaultdict(int))
    for file_number, file_contents in dataset_mms.items():
        state = State.NOT_FOUND
        train_data = TrainData()
        train_positions = {}
        word_after_train = None
        for line_num, row in enumerate(file_contents):
            word = row['maingloss']
            if state == State.DONE_PARSING:
                state = State.NOT_FOUND
                if word_after_train is None:
                    word_after_train = word
                if has_numbers(word_after_train):
                    print(f'WARNING: found number in {word_after_train} after the train (file_number: {file_number})')
                word_after_train = None
            if state == State.NOT_FOUND:
                train_data = TrainData()
                train_detected = False
                if word in ['fa:R-E', 'fa:R-B', 'fa:I-C', 'ICE']:
                    train_type = TrainType(word.removeprefix('fa:'))
                    train_data.set_train_type_mms(train_type, line_num)
                    state = State.TRAIN
                    train_detected = True
                else:
                    state = State.NOT_FOUND
                word_no_dashes = word.replace('-', '')
                regex_matches = re.match(r'.*(?:RE|RB|IC|ICE)$', word_no_dashes) is not None
                should_be_detected = False
                if regex_matches:
                    should_be_detected = True
                    if word.isupper() and len(word) >= 4:
                        should_be_detected = False
                if train_detected != should_be_detected:
                    if train_detected:
                        print(f"WARNING: train type detection wrong: got {word} in file {file_number}")
                    else:
                        print(f"WARNING: train type detection incomplete: got {word} in file {file_number}")
            elif state == State.TRAIN:
                if word.startswith('num:'):
                    state = State.FIRST_NUM
                    train_data.set_num_mms(word.removeprefix('num:'), line_num)
                else:
                    print(f'WARNING: expected num:, got {word} in file {file_number}')
                    state = State.NOT_FOUND
            elif state == State.FIRST_NUM:
                if word.startswith('num:'):
                    state = State.TRAIN_FOUND
                    train_data.set_num_mms(word.removeprefix('num:'), line_num)
                else:
                    state = State.TRAIN_FOUND
                    word_after_train = word

            (state, train_positions) = process_train(state, train_data, train_positions)
        if state in [State.TRAIN, State.FIRST_NUM, State.SECOND_NUM]:
            state = State.TRAIN_FOUND
        # Do it a second time after the loop in case the train is at the end of the file
        (state, train_positions) = process_train(state, train_data, train_positions)



        new_mms_data = []
        for row in dataset_mms[file_number]:
            new_row = row.copy()
            new_mms_data.append(new_row)

        for file_number, train_infos in train_positions.items():
            for train_info in train_infos:
                (type_line, old_type, num1_line, num2_line, old_num) = train_info
                old_train = (old_type, old_num)
                if old_train not in train_mappings[file_number]:
                    new_num1 = None
                    new_num2 = None
                    # TODO: change the train type if number of lines starts the same
                    assert type_line is not None
                    if num2_line is not None:
                        new_type = pick_random_train_type(exclude={TrainType.ICE})
                    else:
                        new_type = pick_random_train_type()
                    if new_type == TrainType.ICE:
                        assert num2_line is None
                        new_num1 = random.randrange(1000, 100000)
                    else:
                        if num1_line is not None and num2_line is None:
                            new_num1 = random.choice(tuple(single_sign_numbers))
                        elif num1_line is not None and num2_line is not None:
                            new_num1 = random.randrange(1, 10)
                        if num2_line is not None:
                            new_num2 = random.choice([20, 30, 40, 50, 60, 70, 80, 90])
                    new_train = (new_type, new_num1, new_num2)
                    train_mappings[file_number][old_train] = new_train

                train_counts[file_number][old_train] += 1
                new_train = train_mappings[file_number][old_train]

                (new_type, new_num1, new_num2) = new_train
                new_type_str = mms_train_type_to_str(new_type)
                new_mms_data[type_line]['maingloss'] = new_type_str

                if new_num1 is not None:
                    new_num1 = mms_train_num_to_str(new_num1, new_num2)
                    new_mms_data[num1_line]['maingloss'] = f'num:{new_num1}'
                if new_num2 is not None:
                    if num2_line < len(new_mms_data):
                        new_mms_data[num2_line]['maingloss'] = f'num:{new_num2}'
                    else:
                        print(f'WARNING: num2_line {num2_line} not found in file {file_number}')
                        for line_num, row in enumerate(new_mms_data):
                            print(f'line {line_num}: {row["maingloss"]}')

        result_mms[file_number] = new_mms_data

    print("train_mappings are", train_mappings)


    result_text = {}
    for folder_name, file_contents in dataset_text.items():
        train_name_pattern = r'\b((RE|RB|IC|ICE)\s*((?:\d-*){1,10}))\b'
        train_per_file = []
        all_distinct_trains_in_file = set()
        for (start_time, end_time, sentence, _number) in file_contents:
            matches = re.findall(train_name_pattern, sentence)
            for (whole_train, train_type, num) in matches:
                train_data = TrainData()
                train_data.set_num_text(num)
                train_type = TrainType[train_type]
                train_data.set_train_type_text(train_type)
                if not train_data.is_valid():
                    continue
                num_int = train_data.get_num_int()
                assembled_train = assemble_train_text(train_type, num_int)
                assert assembled_train == whole_train, f'ERROR: assembled_train is {assembled_train}, whole_train is {whole_train}'
                all_distinct_trains_in_file.add((train_type, whole_train, num_int))

        whole_train_mapping = {}
        train_mapping_str_to_tuple = {}
        for (old_type, old_whole_train, old_num) in all_distinct_trains_in_file:
            train_mapping = train_mappings[folder_name]
            if (old_type, old_num) not in train_mapping:
                print(f'WARNING: old_type {old_type} old_num {old_num} not found in train_mapping (file: {folder_name})')
                continue
            (new_type, new_num1, new_num2) = train_mapping[(old_type, old_num)]
            if new_num2 is not None:
                new_num1 += new_num2
            new_whole_train = assemble_train_text(new_type, new_num1)
            whole_train_mapping[old_whole_train] = new_whole_train
            train_mapping_str_to_tuple[old_whole_train] = (old_type, old_num)


        new_text_data = []
        replaced_counts = defaultdict(int)
        for (start_time, end_time, sentence, number) in file_contents:
            sentence, replaced_counts_per_line = replace_multiple(sentence, whole_train_mapping)
            for old_whole_train, counts in replaced_counts_per_line.items():
                old_whole_train_tuple = train_mapping_str_to_tuple[old_whole_train]
                replaced_counts[old_whole_train_tuple] += counts
            new_text_data.append((start_time, end_time, sentence, number))
        result_text[folder_name] = new_text_data

        for train, count in train_counts[folder_name].items():
            replaced_count = replaced_counts[train]
            if replaced_count != count:
                print(f'WARNING: replaced_count in file {folder_name} should be {count} but was {replaced_count}, trying to replace {train}')

    return (result_text, result_mms)



if __name__ == "__main__":
    from .dataset import *
    dataset_text = read_dataset_text()
    dataset_mms = read_dataset_mms()

    result_text, result_mms = replace_train_entities(dataset_text, dataset_mms)
    write_dataset_text(dataset_text, result_text, main_folder = 'modified/train_name/text')
    write_dataset_mms(dataset_mms, result_mms, main_folder = 'modified/train_name/mms')
