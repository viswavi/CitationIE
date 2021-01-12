import argparse
import os

def sanitize_error_file(error_file_contents):
    special_characters = ["{+{", "<+<", "{-{", "<-<", "}+}", ">+>", "}-}", ">->"]
    for w in special_characters:
        error_file_contents = error_file_contents.replace(w, "")
    return error_file_contents

def process_files(file_names, input_directory, output_directory):
    for f in file_names:
        input_contents = open(os.path.join(input_directory, f)).read()
        output_contents = sanitize_error_file(input_contents)
        with open(os.path.join(output_directory, f), 'w') as outfile:
            outfile.write(output_contents)
    print(f"Processed {len(file_names)} files")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory',
                            type=str,
                            default=None,
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--output_directory',
                            type=str,
                            default=None,
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    args = parser.parse_args()

    os.makedirs(args.output_directory)
    file_names = os.listdir(args.input_directory)
    if len(file_names) == 0:
        print(f"No file names found in directory f{args.input_directory}!")
        breakpoint()
    process_files(file_names, args.input_directory, args.output_directory)

if __name__ == "__main__":
    main()