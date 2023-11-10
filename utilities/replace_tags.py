import argparse
import os


def replace_tag_in_file(file_path, old_name, new_name):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the content by commas to get the tags
    tags = content.split(',')

    # Replace the old_name tag with new_name if it exists
    tags = [new_name if tag.strip() == old_name else tag for tag in tags]

    # Join the tags back with commas
    content = ','.join(tags)

    with open(file_path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Replace tags in txt files.")
    parser.add_argument("--src_dir", required=True,
                        help="Source directory containing txt files.")
    parser.add_argument("--old_name", required=True,
                        help="Old tag name to be replaced.")
    parser.add_argument("--new_name", required=True,
                        help="New tag name to replace the old one.")
    parser.add_argument("--ext", default='.txt',
                        help="Extension of the files to replace tags.")

    args = parser.parse_args()

    # Walk through the src_dir and find all txt files
    for dirpath, dirnames, filenames in os.walk(args.src_dir):
        for filename in filenames:
            if filename.endswith(args.ext):
                file_path = os.path.join(dirpath, filename)
                replace_tag_in_file(file_path, args.old_name, args.new_name)


if __name__ == "__main__":
    main()
