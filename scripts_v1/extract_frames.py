import argparse

from anime2sd import extract_and_remove_similar


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default='.',
                        help="directory containing source files")
    parser.add_argument("--dst_dir", default='.',
                        help="directory to save output files")
    parser.add_argument("--prefix", default='', help="output file prefix")
    parser.add_argument("--ep_init",
                        type=int,
                        default=1,
                        help="episode number to start with")
    parser.add_argument(
        "--similar_thresh",
        type=float,
        default=0.985,
        help="cosine similarity threshold for image duplicate detection")
    parser.add_argument("--no-remove-similar",
                        action="store_true",
                        help="flag to not remove similar images")
    args = parser.parse_args()

    # Process the files
    extract_and_remove_similar(args.src_dir, args.dst_dir, args.prefix,
                               args.ep_init, thresh=args.similar_thresh,
                               to_remove_similar=not args.no_remove_similar)
