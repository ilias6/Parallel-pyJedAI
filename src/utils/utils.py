'''
Utility functions
'''


def insert_to_dict(dictionary: dict, key: any, value: any, value_type: any) -> dict:

    if value_type in ('int' or 'str' or 'float'):
        dictionary[key] = value
    elif value_type == 'list':
        if key not in dictionary.keys():
            dictionary[key] = []
        dictionary[key].append(value)
    elif value_type == 'set':
        if key not in dictionary.keys():
            dictionary[key] = {}
        dictionary[key].add(value)
    else:
        print("Not supported Value Type")
