import os
import argparse

from collections import Counter


def count_tags_in_directory(directory):
    # Counter object to hold the tags and their counts
    tags_counter = Counter()

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .txt file
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    # Read the content of the file
                    content = f.read()
                    # Split the content by commas to get the tags
                    tags = content.split(',')
                    # Remove leading and trailing whitespaces from each tag
                    tags = [tag.strip() for tag in tags]
                    # Update the counter with the tags
                    tags_counter.update(tags)

    return tags_counter


def save_tags_count_to_file(tags_counter, output_file):
    # Sort the tags by frequency in descending order
    sorted_tags = sorted(
        tags_counter.items(), key=lambda x: x[1], reverse=True)

    # Write the sorted tags and their counts to the output file
    with open(output_file, 'w') as f:
        for tag, count in sorted_tags:
            f.write(f"{tag}: {count}\n")


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Count and sort tags from text files in a directory.')

    # Add arguments
    parser.add_argument('--directory', type=str,
                        help='Path to the directory to search for .txt files.')
    parser.add_argument('--output_file', type=str,
                        default='output.txt', help='Path to the output file.')

    # Parse the arguments
    args = parser.parse_args()

    # Count the tags in the specified directory
    tags_counter = count_tags_in_directory(args.directory)

    # Save the tags count to the specified output file
    save_tags_count_to_file(tags_counter, args.output_file)
