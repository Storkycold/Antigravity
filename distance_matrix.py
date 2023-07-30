import os
import util
import trimesh
import numpy as np
import pandas as pd

def distance_matrix(source_root='../week1_sample_ver2'):
    '''
    Make distance matrix of human's meshes.

    :param source_root: str
        The parent path of human mesh folders.
    :param remove_ids: str list
        If there is some error file, please pass the human ids which you want to remove.
    :return distance_matrix: pandas.DataFrame
        Distance matrix of the human meshes.
    '''

    util.log.info('Start to Make a Distance Matrix.')
    subject_names = [os.path.basename(f) for f in util.get_dirs(source_root)]

    distance_matrix = pd.DataFrame(data=np.zeros((len(subject_names), len(subject_names))), index=subject_names,
                                   columns=subject_names)

    for subject_name in subject_names:
        util.log.info('--------------------------------------------------------')
        util.log.info(f'subject_name: {subject_name}')
        affine_file = os.path.join(source_root, subject_name, subject_name + '_affine_trans.stl')
        affine_mesh = trimesh.load_mesh(affine_file)
        affine_faces_edges_len = affine_mesh.edges_unique_length[affine_mesh.faces_unique_edges]

        reg_files = util.get_files(os.path.join(source_root, subject_name), file_type='stl', suffix='0.0001')

        for reg_file in reg_files:
            reg_name = reg_file[-21: -11]
            util.log.info(f'reg_name: {reg_name}')
            try:
                reg_mesh = trimesh.load_mesh(reg_file)
            except:
                continue
            reg_faces_edges_len = reg_mesh.edges_unique_length[reg_mesh.faces_unique_edges]
            distance_edges = np.abs(affine_faces_edges_len - reg_faces_edges_len)
            distance = np.sum(distance_edges) / (distance_edges.shape[0] * distance_edges.shape[1])
            distance_matrix.loc[subject_name, reg_name] = distance
            distance_matrix.loc[reg_name, subject_name] = distance

    distance_matrix.to_csv(os.path.join(source_root, 'distance_matrix_baseline.csv'))

    while True:
        max_count = 0
        removed_index = None
    
        for i, row_name in enumerate(distance_matrix.index):
            zero_count = (distance_matrix.loc[row_name] == 0).sum()

            if zero_count > max_count and zero_count > 1:
                max_count = zero_count
                removed_index = row_name
       
        if removed_index is not None:
            distance_matrix.drop(index=removed_index, columns=removed_index, inplace=True)
        else:
            break

    distance_matrix.to_csv(os.path.join(source_root, 'distance_matrix.csv'))

    util.log.info('End to Make the Distance Matrix.')

    return distance_matrix



