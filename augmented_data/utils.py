import re


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def parse_num(word, folder_name):
    num_str = word.removeprefix('num:')
    try:
        return int(num_str)
    except ValueError:
        print(f'WARNING: expected number, got {num_str} in file {folder_name}')
        return None


def normalize_text_to_mms(text):
    text = text.upper()
    text = text.replace('Ä', 'AE')
    text = text.replace('Ö', 'OE')
    text = text.replace('Ü', 'UE')
    return text


# Example:
# mapping = {"condition1": "", "condition2": "text"}
def replace_multiple(text, mapping):
    if mapping == {}:
        return text, {}
    #print(f"DEBUG: replace_multiple('{text}', {mapping})")
    mapping_escaped = dict((re.escape(k), v) for k, v in mapping.items())
    pattern = re.compile("|".join(mapping_escaped.keys()))
    new_text = pattern.sub(lambda m: mapping_escaped[re.escape(m.group(0))], text)
    replaced_counts = {}
    for k in mapping:
        matches = re.findall(re.escape(k), text)
        replaced_counts[k] = len(matches)
    return new_text, replaced_counts
