import os

def print_directory_tree(root_dir, indent='', ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = set(['__pycache__', '.vscode', '.env'])

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            if item not in ignore_dirs:
                print(indent + '├── ' + item)
                print_directory_tree(item_path, indent + '│   ', ignore_dirs)
        else:
            print(indent + '├── ' + item)


if __name__ == "__main__":
    os.system('clear')
    os.chdir('../../..')
    root = os.getcwd()
    print(root)
    print_directory_tree(root)
