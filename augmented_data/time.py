# Run with: python -m augmented_data.time

import re
import random
from enum import Enum, verify, UNIQUE
from collections import defaultdict
from .utils import has_numbers, replace_multiple

# Exclude 0
single_sign_minutes = set()
for i in range(1, 20):
    single_sign_minutes.add(i)

for i in range(2, 6):
    single_sign_minutes.add(i * 10) # Tens
    single_sign_minutes.add(i * 11) # Repdigits

print(f'single_sign_minutes: {single_sign_minutes}')

class TimeData:
    def __init__(self):
        self.hour_str = None
        self.minute1_str = None
        self.minute2_str = None
        self.hour_int = None
        self.minute_int = None
        self.hour_line = None
        self.minute1_line = None
        self.minute2_line = None
        self.hour_is_separate = None

    def is_valid(self):
        if self.hour_str is None:
            print('ERROR: hour is None')
            return False
        try:
            self.hour_int = int(self.hour_str)
            if self.minute_int is None:
                self.minute_int = 0
        except ValueError:
            print(f'ERROR: hour is not a number: {self.hour_str}')
            return False
        if self.hour_int < 0 or self.hour_int > 23:
            print(f'ERROR: not a valid hour: {self.hour_int}')
            self.hour_int = None
            return False
        if self.minute_int < 0 or self.minute_int > 59:
            print(f'ERROR: not a valid minute: {self.minute_int}')
            self.minute_int = None
            return False
        return True

    def set_hour(self, hour_str, line_num=None):
        if self.hour_str is not None:
            raise Exception('ERROR: hour can only be set once')
        self.hour_str = hour_str
        self.hour_line = line_num

    def set_minute_mms(self, minute_str, line_num):
        if self.minute2_str is not None:
            raise Exception('ERROR: minute was already set')
        if minute_str == '':
            raise Exception('ERROR: minute is empty')
        try:
            minute_int = int(minute_str)
        except ValueError:
            raise Exception(f'ERROR: minute is not a number: {minute_str}')
        if minute_int < 0:
            raise Exception(f'ERROR: minute is negative: {minute_int}')

        if self.minute1_str is None:
            if minute_int not in single_sign_minutes:
                print(f'WARNING: minute1 is not a valid single sign value: {minute_str}')
            self.minute1_str = minute_str
            self.minute1_line = line_num
        else:
            if minute_str not in ['20', '30', '40', '50']:
                print(f'WARNING: minute2 is not a valid value: {minute_str}')
            if self.minute_int not in range(1, 10):
                print(f'WARNING: minute1 is not a valid value: {self.minute1_str}')
            self.minute2_str = minute_str
            self.minute2_line = line_num

        if self.minute_int is None:
            self.minute_int = 0
        self.minute_int += minute_int

    def set_minute_text(self, minute_str):
        if self.minute1_str is not None:
            raise Exception('ERROR: minute was already set')
        if minute_str == '':
            raise Exception('ERROR: minute is empty')
        self.minute1_str = minute_str
        try:
            minute_int = int(minute_str)
        except ValueError:
            raise Exception(f'ERROR: minute is not a number: {minute_str}')

        self.minute_int = minute_int

    def set_hour_is_separate(self, hour_is_separate):
        if self.hour_is_separate is not None:
            raise Exception('ERROR: hour_is_separate can only be set once')
        self.hour_is_separate = hour_is_separate

    def get_line_numbers(self):
        if self.hour_line is None:
            raise Exception('ERROR: hour has not been set yet')
        return (self.hour_line, self.minute1_line, self.minute2_line)

    def get_time_int(self):
        if self.hour_int is None:
            raise Exception('ERROR: hour has not been set yet')
        return (self.hour_int, self.minute_int)

    def get_hour_is_separate(self):
        return self.hour_is_separate

@verify(UNIQUE)
class TimeFormat(Enum):
    UNKNOWN = 0
    WITH_COLON = 1 # 18:20 Uhr
    JUST_HOUR = 2 # 18 Uhr
    HOUR_UHR_MINUTE = 3 # 18 Uhr 30

def assemble_time(time_format, hour_int, minute_int):
    if time_format == TimeFormat.WITH_COLON:
        return f'{hour_int:02d}:{minute_int:02d} Uhr'
    elif time_format == TimeFormat.JUST_HOUR:
        return f'{hour_int} Uhr'
    elif time_format == TimeFormat.HOUR_UHR_MINUTE:
        return f'{hour_int} Uhr {minute_int:02d}'
    else:
        raise Exception(f'ERROR: unknown time format {time_format}')



def replace_time_entities(dataset_text, dataset_mms):
    @verify(UNIQUE)
    class State(Enum):
        NOT_FOUND = 0
        FIRST_NUM = 1
        UHR = 2
        SECOND_NUM = 3
        TIME_FOUND = 4
        DONE_PARSING = 5

    def process_time(state, time_data, time_positions):
        if state == State.TIME_FOUND:
            state = State.DONE_PARSING
            if not time_data.is_valid():
                return (state, time_positions)
            # Here we know that the time is valid
            (hour_line, minute1_line, minute2_line) = time_data.get_line_numbers()
            (hour_int, minute_int) = time_data.get_time_int()
            hour_is_separate = time_data.get_hour_is_separate()
            print(f'Found time: {hour_int}:{minute_int}, lines: {hour_line}, {minute1_line}, {minute2_line} (file_number: {file_number})')
            time_position = (hour_line, minute1_line, minute2_line, hour_int, minute_int, hour_is_separate)
            time_positions_list = time_positions.get(file_number, [])
            time_positions_list.append(time_position)
            time_positions[file_number] = time_positions_list
        return (state, time_positions)

    result_mms = {}
    time_mappings = defaultdict(dict)
    time_counts = defaultdict(lambda: defaultdict(int))
    for file_number, file_contents in dataset_mms.items():
        state = State.NOT_FOUND
        time_data = TimeData()
        time_positions = {}
        word_after_time = None
        for line_num, row in enumerate(file_contents):
            word = row['maingloss']
            if state == State.DONE_PARSING:
                state = State.NOT_FOUND
                if word_after_time is None:
                    word_after_time = word
                if has_numbers(word_after_time):
                    print(f'WARNING: found number in {word_after_time} after the time (file_number: {file_number})')
                word_after_time = None
            if state == State.NOT_FOUND:
                # time_data = recreate object here
                time_data = TimeData()
                if word.startswith('num:'):
                    state = State.FIRST_NUM
                    time_data.set_hour(word.removeprefix('num:'), line_num)
                    time_data.set_hour_is_separate(True)
                elif word.startswith('uhr:'):
                    state = State.UHR
                    time_data.set_hour(word.removeprefix('uhr:'), line_num)
                    time_data.set_hour_is_separate(False)
            elif state == State.FIRST_NUM:
                if word == 'UHR':
                    state = State.UHR
                else:
                    state = State.NOT_FOUND
            elif state == State.UHR:
                if word.startswith('num:'):
                    state = State.SECOND_NUM
                    time_data.set_minute_mms(word.removeprefix('num:'), line_num)
                else:
                    state = State.TIME_FOUND
                    word_after_time = word
            elif state == State.SECOND_NUM:
                if word.startswith('num:'):
                    state = State.TIME_FOUND
                    time_data.set_minute_mms(word.removeprefix('num:'), line_num)
                else:
                    state = State.TIME_FOUND
                    word_after_time = word

            (state, time_positions) = process_time(state, time_data, time_positions)
        if state in [State.UHR, State.SECOND_NUM]:
            state = State.TIME_FOUND
        # Do it a second time after the loop in case the time is at the end of the file
        (state, time_positions) = process_time(state, time_data, time_positions)



        new_mms_data = []
        for row in dataset_mms[file_number]:
            new_row = row.copy()
            new_mms_data.append(new_row)

        for file_number, time_infos in time_positions.items():
            for time_info in time_infos:
                (hour_line, minute1_line, minute2_line, old_hour, old_minute, hour_is_separate) = time_info
                old_time = (old_hour, old_minute)
                if old_time not in time_mappings[file_number]:
                    new_hour = random.randrange(24)
                    new_minute1 = None
                    new_minute2 = None
                    if minute1_line is not None and minute2_line is None:
                        new_minute1 = random.choice(tuple(single_sign_minutes))
                    elif minute1_line is not None and minute2_line is not None:
                        new_minute1 = random.randrange(1, 10)
                    if minute2_line is not None:
                        new_minute2 = random.choice([20, 30, 40, 50])
                    new_time = (new_hour, new_minute1, new_minute2)
                else:
                    new_time = time_mappings[file_number][old_time]

                (new_hour, new_minute1, new_minute2) = new_time
                if hour_is_separate:
                    new_hour_str = f'num:{new_hour}'
                else:
                    new_hour_str = f'uhr:{new_hour}'
                new_mms_data[hour_line]['maingloss'] = new_hour_str
                if new_minute1 is not None:
                    new_mms_data[minute1_line]['maingloss'] = f'num:{new_minute1}'
                if new_minute2 is not None:
                    if minute2_line < len(new_mms_data):
                        new_mms_data[minute2_line]['maingloss'] = f'num:{new_minute2}'
                    else:
                        print(f'WARNING: minute2_line {minute2_line} not found in file {file_number}')
                        for line_num, row in enumerate(new_mms_data):
                            print(f'line {line_num}: {row["maingloss"]}')

                time_mappings[file_number][old_time] = new_time
                time_counts[file_number][old_time] += 1
        result_mms[file_number] = new_mms_data



    result_text = {}
    for folder_name, file in dataset_text.items():
        # find all time of pattern 18 Uhr 30 and 20 Uhr and 12:20 Uhr
        time_pattern = r'\b((\d{1,2}):(\d{2}) Uhr|(\d{1,2}) Uhr(?: (\d{1,2}))?)\b'
        time_per_file = []
        all_distinct_times_in_file = set()
        for (start_time, end_time, sentence, number) in file:
            matches = re.findall(time_pattern, sentence)
            for (whole_time, hour1, minute1, hour2, minute2) in matches:
                time_data = TimeData()
                time_format = TimeFormat.UNKNOWN
                if hour1 != '':
                    assert hour2 == ''
                    assert minute2 == ''
                    time_data.set_hour(hour1)
                    time_data.set_minute_text(minute1)
                    time_format = TimeFormat.WITH_COLON
                else:
                    assert hour2 != ''
                    time_data.set_hour(hour2)
                    if minute2 != '':
                        time_data.set_minute_text(minute2)
                        time_format = TimeFormat.HOUR_UHR_MINUTE
                    else:
                        time_format = TimeFormat.JUST_HOUR
                if not time_data.is_valid():
                    continue
                (hour_int, minute_int) = time_data.get_time_int()
                assembled_time = assemble_time(time_format, hour_int, minute_int)
                assert assembled_time == whole_time, f'ERROR: assembled_time is {assembled_time}, whole_time is {whole_time}'
                all_distinct_times_in_file.add((time_format, whole_time, hour_int, minute_int))

        whole_time_mapping = {}
        time_mapping_str_to_tuple = {}
        for (time_format, old_whole_time, old_hour, old_minute) in all_distinct_times_in_file:
            old_time = (old_hour, old_minute)
            time_mapping = time_mappings[folder_name]
            if old_time not in time_mapping:
                print(f'WARNING: old_time {old_time} not found in time_mapping (file: {folder_name})')
                continue
            (new_hour, new_minute1, new_minute2) = time_mapping[old_time]
            if new_minute2 is not None:
                new_minute1 += new_minute2
            if new_minute1 is None and new_minute2 is None and time_format != TimeFormat.JUST_HOUR:
                print(f'WARNING: new_minute1 and new_minute2 are None but time_format is {time_format} (file: {folder_name})')
                continue
            new_whole_time = assemble_time(time_format, new_hour, new_minute1)
            whole_time_mapping[old_whole_time] = new_whole_time
            time_mapping_str_to_tuple[old_whole_time] = old_time


        new_text_data = []
        replaced_counts = defaultdict(int)
        for (start_time, end_time, sentence, number) in file:
            sentence, replaced_counts_per_line = replace_multiple(sentence, whole_time_mapping)
            for old_time, counts in replaced_counts_per_line.items():
                old_time_tuple = time_mapping_str_to_tuple[old_time]
                replaced_counts[old_time_tuple] += counts
            new_text_data.append((start_time, end_time, sentence, number))
        result_text[folder_name] = new_text_data

        for time, count in time_counts[folder_name].items():
            replaced_count = replaced_counts[time]
            if replaced_count != count:
                print(f'WARNING: replaced_count in file {folder_name} should be {count} but was {replaced_count}, trying to replace {time}')

    return (result_text, result_mms)



if __name__ == "__main__":
    from .dataset import *
    dataset_text = read_dataset_text()
    dataset_mms = read_dataset_mms()

    result_text, result_mms = replace_time_entities(dataset_text, dataset_mms)
    write_dataset_text(result_text, main_folder = 'modified/time/text')
    write_dataset_mms(result_mms, main_folder = 'modified/time/mms')
