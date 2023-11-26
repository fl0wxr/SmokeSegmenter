import re
import os


def remove_blacklisted_files(BLACKLIST_FP = '../blacklisted_instances.list'):

    with open(BLACKLIST_FP, 'r') as file:
        blacklist_str = file.read()
    blacklist_paths = re.split(pattern = '\n|, ', string = blacklist_str)
    if blacklist_paths[-1] == '':
        blacklist_paths = blacklist_paths[:-1]

    for fp in blacklist_paths:
        os.remove(fp)


if __name__ == '__main__':

    remove_trigger = input('W: All blacklisted files will be deleted. Proceed? Yes [Y] No [N]\n> ')
    while remove_trigger not in {'Y', 'N'}:
        remove_trigger = input('Select a valid input')

    remove_trigger = remove_trigger == 'Y'

    if remove_trigger:
        remove_blacklisted_files()
