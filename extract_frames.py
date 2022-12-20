import os
import glob
import argparse
import shlex


def extract_star_match(pattern: str, s: str) -> str:
    # This simple function works in my case but should be imrpoved
    # Split the pattern into two parts around the "*" character
    parts = pattern.split("*")
    # Check if the string starts with the first part of the pattern
    if s.startswith(parts[0]):
        # Check if the string ends with the second part of the pattern
        if s.endswith(parts[1]):
            # Return the part of the string
            # in between the two parts of the pattern
            return s[len(parts[0]):len(s)-len(parts[1])]
    # If the string does not match the pattern, return an empty string
    return ""


def process_files(src_dir, dst_dir, prefix, pattern):
    # Find all files in the specified source directory that match the pattern
    pattern = os.path.join(src_dir, pattern)
    files = glob.glob(pattern)
    # Dealing with brackets
    pattern = pattern.replace('[[]', '[').replace('[]]', ']')
    print(pattern)
    print(files)

    # Loop through each file
    for file in files:
        # Extract the episode number from the file name
        ep = extract_star_match(pattern, file)

        # Create the output directory
        os.makedirs(os.path.join(dst_dir, f"EP{ep}"), exist_ok=True)
        file_pattern = os.path.join(
            dst_dir, f'EP{ep}', f'{prefix}{ep}_%d.png')

        # Run ffmpeg on the file, saving the output to the output directory
        ffmpeg_command = \
            f"ffmpeg -hwaccel cuda -i {shlex.quote(file)} -filter:v "\
            "'mpdecimate=hi=64*200:lo=64*50:"\
            "frac=0.33,setpts=N/FRAME_RATE/TB' "\
            f"-qscale:v 1 -qmin 1 -c:a copy {file_pattern}"
        print(ffmpeg_command)
        os.system(ffmpeg_command)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default='.',
                        help="directory containing source files")
    parser.add_argument("--dst_dir", default=',',
                        help="directory to save output files")
    parser.add_argument("--prefix", required=True, help="output file prefix")
    parser.add_argument("--pattern", required=True,
                        help="pattern for the files")
    args = parser.parse_args()

    # Process the files
    process_files(args.src_dir, args.dst_dir, args.prefix, args.pattern)
