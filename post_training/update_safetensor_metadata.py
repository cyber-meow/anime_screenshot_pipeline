import argparse
import json
import struct
import os
import shutil


def read_metadata(safetensors_file_path):
    with open(safetensors_file_path, 'rb') as file:
        length_of_header = struct.unpack('<Q', file.read(8))[0]
        header_data = file.read(length_of_header).decode('utf-8')
        # Return the header and the offset to the data
        return json.loads(header_data), 8 + length_of_header


def write_updated_file(original_file_path, updated_metadata, data_offset):
    temp_file_path = original_file_path + '.temp'
    with (open(original_file_path, 'rb') as original_file,
            open(temp_file_path, 'wb') as temp_file):
        # Write updated metadata to temp file
        header_str = json.dumps(updated_metadata)
        header_bytes = header_str.encode('utf-8')
        temp_file.write(struct.pack('<Q', len(header_bytes)))
        temp_file.write(header_bytes)

        # Seek to the beginning of the data in the original file
        original_file.seek(data_offset)

        # Copy data from original file to temp file
        shutil.copyfileobj(original_file, temp_file)

    # Replace original file with temp file
    os.replace(temp_file_path, original_file_path)


def update_metadata(text_file_path, safetensors_file_path):
    metadata, data_offset = read_metadata(safetensors_file_path)
    # Remove 'cm_wildcard' if it exists; ignore error if it doesn't
    metadata['__metadata__'].pop('cm_wildcard', None)
    metadata['__metadata__'].pop('ss_dataset_dirs', None)
    metadata['__metadata__'].pop('ss_tag_frequency', None)

    if text_file_path is not None and os.path.exists(text_file_path):
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            text_content = [line.strip() for line in text_file.readlines()]

            updated_metadata = {'cm_wildcard': json.dumps(text_content)}
            updated_metadata.update(metadata['__metadata__'])

            # Update the '__metadata__' field in the original dictionary
            metadata['__metadata__'] = updated_metadata
    write_updated_file(safetensors_file_path, metadata, data_offset)


def main():
    parser = argparse.ArgumentParser(
        description='Update Safetensors metadata.')
    parser.add_argument('--wildcard_file', default=None,
                        help='Path to the wildcard file.')
    parser.add_argument('--safetensors_file_or_folder',
                        help='Path to the Safetensors file or folder.')
    args = parser.parse_args()

    if os.path.isdir(args.safetensors_file_or_folder):
        for root, dirs, files in os.walk(args.safetensors_file_or_folder):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensors_file_path = os.path.join(root, file)
                    update_metadata(args.wildcard_file, safetensors_file_path)
    else:
        update_metadata(args.wildcard_file, args.safetensors_file_or_folder)


if __name__ == '__main__':
    main()
