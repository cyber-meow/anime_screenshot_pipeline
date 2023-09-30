import os
import sys


if __name__ == '__main__':

    dir1 = sys.argv[1]
    image_extensions = ['.png', '.jpg', '.jpeg']
    file_scores = []
    for filename in os.listdir(dir1):
        filename_noext, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            tag_file = filename + '.tags'
            with open(os.path.join(dir1, tag_file), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('score:'):
                    score = int(line.lstrip('score:').strip())
                    file_scores.append((score, filename))
    file_scores = list(reversed(sorted(file_scores)))
    retain = int(sys.argv[2])
    dst_folder = os.path.join(dir1, 'high_score')
    os.makedirs(dst_folder, exist_ok=True)
    for (_, file) in file_scores[:retain]:
        os.rename(os.path.join(dir1, file),
                  os.path.join(dst_folder, file))
        os.rename(os.path.join(dir1, file + '.tags'),
                  os.path.join(dst_folder, file + '.tags'))
