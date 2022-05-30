'''
Utility functions
'''


def insert_to_dict(dictionary: dict, key: any, value: any, value_type: any = 'str') -> dict:

    if value_type in ('int' or 'str' or 'float'):
        dictionary.setdefault(key, value)
    elif value_type == 'list':
        dictionary.setdefault(key, [])
        dictionary[key].append(value)
    elif value_type == 'set':
        dictionary.setdefault(key, {})
        dictionary[key].add(value)
    else:
        print("Not supported Value Type")

