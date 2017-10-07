import sys, os, argparse, logging, subprocess, shutil

def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False


mode = 'tokenize' or 'normalize'
num_threads = 4
input_file
output_file
log_path

def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Script being executed: %s'%__file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s"%(input_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:  
        with open(output_file) as fin:
            for line in fin:
                fout.write(line.replace('\r', ' ').strip() + '\n')  # delete \r

    cmd = "cat %s | node preprocess_latex.js %s > %s "%(temp_file, parameters.mode, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    if is_ascii(token):
                        tokens_out.append(token)
                fout.write(' '.join(tokens_out)+'\n')
    os.remove(temp_file)