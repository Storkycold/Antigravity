'''
main.py
'''

import util
import landmark as lm
import registration as reg
import argparse
import os
import distance_matrix as dm
import matrix_tessellation as mt
import mlp
import multiprocessing as mp 
from multiprocessing import Pool
from multiprocessing import Lock, Manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', '-log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level. Choose one from ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']. Default value is 'INFO'")
    parser.add_argument('--subject_path', '-subject', type=str, default='../week1_sample_ver2', help='The parent path of the HUMAN_ID folders.')
    parser.add_argument('--upsampled_path', type=str, default='../Upsampled_6th_20s_women', help='The parent path of the upsampled mesh folders.')
    parser.add_argument('--core', '-c', type=int, default=4, help='The number of cores to be used for multiprocessing. Default value is 1.')
    args = parser.parse_args()

    # Set the logger
    util.set_logger(args.log_level)

    # Get the subject directories
    subject_dirs = util.get_dirs(path=args.subject_path)
    util.log.info(f'Subject number: {len(subject_dirs)}')
    util.log.info(f'Subject list: {[os.path.basename(s) for s in subject_dirs]}')

    # Find the corresponding vertex index for the landmark position.
    lm.make_lm2v(subject_dirs)
    # Create a list of input parameters to be passed to the registration function.
    multi_input_list = []
    for i, target_dir in enumerate(subject_dirs):
        for source_dir in subject_dirs[i + 1:]:
            input_list = [source_dir, target_dir, args.upsampled_path, 0.01, False]
            multi_input_list += [input_list]

    with mp.Manager() as manager:
        lock = manager.Lock()
        shared_count = manager.Value(int, 0)
        total_count = manager.Value(int, len(multi_input_list))
    # util.total_process = len(multi_input_list)
        with mp.Pool(processes=args.core) as pool: 
            error_files = pool.starmap(reg.multiprocessing_registration, [[shared_count, total_count, lock] + i for i in multi_input_list])

    # with mp.pool.Pool(initializer=init_worker, initargs=(len(multi_input_list), 0, ), processes=args.core) as p:
    #     error_files = p.starmap(multiprocessing_registration, multi_input_list)

    # for error_file in error_files:
    #     util.log.info(f'Registration error files: {error_file}')

    # Create a square distance matrix for all subjects.
    # distance_matrix = dm.distance_matrix(args.subject_path)

    # # Create a tessellation map using the distance matrix.
    # tessellation = mt.tessellation(args.subject_path)

    # # Train an MLP model to learn the relationship between independent variables and the tessellation map.
    # mlp.train(args.subject_path)

# def add_to_value(addend, value, lock):
#     with lock:
#         value.value += addend

# if __name__ == '__main__':
#     with multiprocessing.Manager() as manager:
#         lock = manager.Lock()
#         value = manager.Value(float, 0.0)
#         with multiprocessing.Pool(2) as pool:
#             pool.starmap(add_to_value,
#                          [(float(i), value, lock) for i in range(100)])
#         print(value.value)