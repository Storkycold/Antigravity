'''
seg_icp.py
'''

import time
import trimesh
from trimesh import intersections, viewer, boolean
from multiprocessing import Pool, freeze_support
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from numpyencoder import NumpyEncoder
import numpy as np
import matplotlib.pyplot as plt
# import pyVHACD
import pyvista as pv
from pyvista import examples
import landmark as lm
from shapely import ops
import trimesh.registration
import pyglet
import util
import viewer
import networkx as nx
import os
import json
import logging


def point_above_plane(point, plane_normal, plane_point):
    '''
    Check if a point is above a certain plane.

    :param point: (3, ) float            --  Point you want to check.
    :param plane_normal: (3, ) float     --  Normal vector of the certain plane.
    :param plane_point: (3, ) float      --  A Point lies on a certain plane.

    :return: bool                        --  If the point is above a certain plane, return true.
    '''

    distance = np.dot(plane_normal, point - plane_point)
    return distance >= 0


def triangle_above_plane(triangle_vertices, plane_normal, plane_point):
    '''
    Check if a triangle face is above a certain plane.

    :param triangle_vertices: (3, 3) float       --  The vertices of a triangle.
    :param plane_normal: (3, ) float             --  Normal vector of the certain plane.
    :param plane_point: (3, ) float              --  A Point lies on a certain plane.

    :return: bool                                --  If the triangle is above a certain plane, return true.
    '''

    for tv in triangle_vertices:
        if not point_above_plane(tv, plane_normal, plane_point):
            return False
    return True


def triangle_intersect_plane(mesh, plane_normal, plane_point):
    '''
    Find the intersecting faces when a mesh and a plane intersect.

    :param mesh: Trimesh
    :param plane_normal: (3, ) float         --  Normal vector of the certain plane.
    :param plane_point: (3, ) float          --  A Point lies on a certain plane.

    :return: list                            --  List of the face indices which are contain intersecting line.
    '''
    _, intersect_faces = intersections.mesh_plane(mesh, plane_normal, plane_point, return_faces=True)
    return list(intersect_faces)


def segmentation(mesh, lm_dict, seg_num=6):
    '''
    Mesh segmentation with landmark.
    Default segmentation is [head, body, right_arm, left_arm, right_leg, left_leg].
    If you set seg_num=7, then segmentation result is
    [head, upper_body, lower_body, right_arm, left_arm, right_leg, left_leg]

    I recommend that you set seg_num=6 when using ICP after segmentation; otherwise, set seg_num=7.

    :param mesh: Trimesh
    :param lm_dict: dict      -- Dictionary of lm2v
    :param seg_num: int       -- The number of segmentation. Choose 6 or 7.

    :return: list             -- List of the segmentation face indices list.
    '''

    # Set the landmarks to be used.
    # 목뒷점 : landmark 6번
    lm6_v = mesh.vertices[int(lm_dict['6'])]
    # 샅앞점 : landmark 48번
    lm48_v = mesh.vertices[int(lm_dict['48'])]
    # 겨드랑점(오른) : landmark 9번
    lm9_v = mesh.vertices[int(lm_dict['9'])]
    # 겨드랑점(왼) : landmark 10번
    lm10_v = mesh.vertices[int(lm_dict['10'])]
    # 볼기고랑점(오른) : landmark 25번
    lm25_v = mesh.vertices[int(lm_dict['25'])]
    # 볼기고랑점(왼) : landmark 60번
    lm60_v = mesh.vertices[int(lm_dict['60'])]
    # 넙다리 가운데점(오른) : landmark 38번
    lm38_v = mesh.vertices[int(lm_dict['38'])]
    # 넙다리 가운데점(왼) : landmark 61번
    lm61_v = mesh.vertices[int(lm_dict['61'])]
    # 손목 안쪽점(오른) : landmark 30번
    lm30_v = mesh.vertices[int(lm_dict['30'])]
    # 손목 안쪽점(왼) : landmark 57번
    lm57_v = mesh.vertices[int(lm_dict['57'])]
    # 미드리프수준점 : landmark 35번
    lm35_v = mesh.vertices[int(lm_dict['35'])]

    # All faces indices
    all_faces = set({i for i in range(len(mesh.faces))})

    # Find head
    head_faces = []
    for i, f in enumerate(mesh.faces):
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        if triangle_above_plane([v0, v1, v2], plane_normal=[0.0, 1.0, 0.0], plane_point=lm6_v):
            head_faces += [i]

    faces_except_head = all_faces - set(head_faces)

    # Find right and left legs
    legs_faces = []
    for f_idx in faces_except_head:
        f = mesh.faces[f_idx]
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        if triangle_above_plane([v0, v1, v2], plane_normal=[0.0, -1.0, 0.0], plane_point=lm25_v):
            legs_faces += [f_idx]

    right_leg_faces = []
    left_leg_faces = []
    for f_idx in legs_faces:
        f = mesh.faces[f_idx]
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        if triangle_above_plane([v0, v1, v2], plane_normal=[-1.0, 0.0, 0.0], plane_point=lm48_v):
            if triangle_above_plane([v0, v1, v2], plane_normal=[1.0, 0.0, 0.0], plane_point=lm30_v):
                right_leg_faces += [f_idx]
        else:
            if triangle_above_plane([v0, v1, v2], plane_normal=[-1.0, 0.0, 0.0], plane_point=lm57_v):
                left_leg_faces += [f_idx]

    faces_except_head_leg = faces_except_head - set(right_leg_faces) - set(left_leg_faces)

    # Find right and left arms
    right_arms_faces = []
    for f_idx in faces_except_head_leg:
        f = mesh.faces[f_idx]
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        if triangle_above_plane([v0, v1, v2], plane_normal=[-1.0, 0.0, 0.0], plane_point=lm9_v):
            right_arms_faces += [f_idx]

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[right_arms_faces])
    s = new_mesh.split(only_watertight=False)
    biggest_split = s[0]
    for m in s:
        if len(m.faces) > len(biggest_split.faces):
            biggest_split = m

    new_right_arms_faces = []
    query = trimesh.proximity.ProximityQuery(biggest_split)
    for f_idx in right_arms_faces:
        closest_point = biggest_split.vertices[query.vertex([mesh.vertices[mesh.faces[f_idx][0]]])[1]]
        if trimesh.util.euclidean(mesh.vertices[mesh.faces[f_idx][0]], closest_point) < 0.1:
            new_right_arms_faces += [f_idx]
    right_arms_faces = new_right_arms_faces

    faces_except_head_leg_rarm = faces_except_head_leg - set(right_arms_faces)

    left_arms_faces = []
    for f_idx in faces_except_head_leg_rarm:
        f = mesh.faces[f_idx]
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        if triangle_above_plane([v0, v1, v2], plane_normal=[1.0, 0.0, 0.0], plane_point=lm10_v):
            left_arms_faces += [f_idx]

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[left_arms_faces])
    s = new_mesh.split(only_watertight=False)
    biggest_split = s[0]
    for m in s:
        if len(m.faces) > len(biggest_split.faces):
            biggest_split = m

    new_left_arms_faces = []
    query = trimesh.proximity.ProximityQuery(biggest_split)
    for f_idx in left_arms_faces:
        closest_point = biggest_split.vertices[query.vertex([mesh.vertices[mesh.faces[f_idx][0]]])[1]]
        if trimesh.util.euclidean(mesh.vertices[mesh.faces[f_idx][0]], closest_point) < 0.1:
            new_left_arms_faces += [f_idx]
    left_arms_faces = new_left_arms_faces

    # Find body
    body_faces = list(faces_except_head_leg_rarm - set(left_arms_faces))

    if seg_num == 6:
        segmentations = [head_faces, body_faces, right_arms_faces, left_arms_faces, right_leg_faces, left_leg_faces]
    else:
        upper_body_faces = []
        lower_body_faces = []
        for f_idx in body_faces:
            f = mesh.faces[f_idx]
            v0 = mesh.vertices[f[0]]
            v1 = mesh.vertices[f[1]]
            v2 = mesh.vertices[f[2]]

            if triangle_above_plane([v0, v1, v2], plane_normal=[0.0, 1.0, 0.0], plane_point=lm35_v):
                upper_body_faces.append(f_idx)
            else:
                lower_body_faces.append(f_idx)

        segmentations = [head_faces, upper_body_faces, lower_body_faces, right_arms_faces, left_arms_faces,
                         right_leg_faces,
                         left_leg_faces]

    return segmentations


def seg_icp(source_mesh, target_mesh, ss, ts):
    '''
    Apply the ICP (Iterative Closest Point) algorithm to the segmented source and target meshes.

    :param source_mesh: Trimesh
    :param target_mesh: Trimesh
    :param ss: list                 -- List of the indices of the segmented source mesh faces.
    :param ts: list                 -- List of the indices of the segmented target mesh faces.

    :return s_m_vidx: list          -- The list of vertices that make up the each segmented mesh.
    :return icp_vertices: list      -- The list of ICP vertices for each segmented mesh.
    '''

    s_m_vidx = list()
    for sf_idxs in ss:
        s_m_vidx.extend(source_mesh.faces[sf_idxs])
    s_m_vidx = sorted(list(set(s_m_vidx)))

    t_m_vidx = list()
    for tf_idxs in ts:
        t_m_vidx.extend(target_mesh.faces[tf_idxs])
    t_m_vidx = sorted(list(set(t_m_vidx)))

    # icp
    procrustes_dict = {'reflection': False, 'translation': True, 'scale': True, 'return_cost': True}
    translation_mat4, icp_vertices, _ = trimesh.registration.icp(source_mesh.vertices[s_m_vidx], target_mesh.vertices[t_m_vidx], initial=None,
                                                                 threshold=1e-05, max_iterations=50, **procrustes_dict)
    return s_m_vidx, icp_vertices


def seg_nricp(source_mesh, target_mesh, ss, ts):
    '''
    Apply the NRICP (Nonrigid Iterative Closest Point Algorithm) algorithm to the segmented source and target meshes.

    :param source_mesh: Trimesh
    :param target_mesh: Trimesh
    :param ss: list                   -- List of the indices of the segmented source mesh faces.
    :param ts: list                   -- List of the indices of the segmented target mesh faces.

    :return s_m_vidx: list            -- The list of vertices that make up the each segmented mesh.
    :return nricp_vertices: list      -- The list of NRICP vertices for each segmented mesh
    '''

    s_m_vidx = list()
    for sf_idxs in ss:
        s_m_vidx.extend(source_mesh.faces[sf_idxs])
    s_m_vidx = sorted(list(set(s_m_vidx)))
    s_m_f = []
    for sf_idxs in ss:
        raw_face = source_mesh.faces[sf_idxs]
        new_face = [s_m_vidx.index(raw_face[0]), s_m_vidx.index(raw_face[1]), s_m_vidx.index(raw_face[2])]
        s_m_f += [new_face]

    s_m = trimesh.Trimesh(vertices=source_mesh.vertices[s_m_vidx], faces=s_m_f, process=False)
    t_m_vidx = list()
    for tf_idxs in ts:
        t_m_vidx.extend(target_mesh.faces[tf_idxs])
    t_m_vidx = sorted(list(set(t_m_vidx)))
    t_m_f = []
    for tf_idxs in ts:
        raw_face = target_mesh.faces[tf_idxs]
        new_face = [t_m_vidx.index(raw_face[0]), t_m_vidx.index(raw_face[1]), t_m_vidx.index(raw_face[2])]
        t_m_f += [new_face]

    t_m = trimesh.Trimesh(vertices=target_mesh.vertices[t_m_vidx], faces=t_m_f, process=False)
    # nricp
    nricp_vertices = trimesh.registration.nricp_amberg(s_m, t_m, steps=None, eps=0.01,
                                                       gamma=1,
                                                       distance_threshold=0.1, return_records=False,
                                                       use_faces=True,
                                                       use_vertex_normals=True, neighbors_count=8)

    return s_m_vidx, nricp_vertices


def make_correspondence_multi(shared_count, total_count, lock, source_dir, target_dir, version='icp', visual=False):
    '''
    Find the vertex correspondence between the source mesh and the target mesh, and save the registration
    result of the source mesh.

    Notice: If you use 'Window', please set freeze_support() before running this function for multiprocessing.

    :param shared_count: int     -- The current progress as a number out of the total progress.
    :param total_count: int      -- The number of total progress.
    :param lock:
    :param source_dir: str       -- The path of the source directory.
    :param target_dir: str       -- The path of the target directory.
    :param version: str          -- Choose 'icp' or 'nricp'
    :param visual: bool          -- If you want to visualize the registration result, set this value 'True'
    '''

    c_proc = mp.current_process()
    proc_log = logging.getLogger(c_proc.name)

    if proc_log.hasHandlers():
        proc_log.handlers.clear()

    proc_log.setLevel('INFO')
    formatter = logging.Formatter("[%(asctime)s] | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    proc_log.addHandler(stream_handler)

    log_prefix = f'Process: {c_proc.name} | '
    with lock:
        shared_count.value += 1
        proc_log.info(log_prefix + f'Make correspondence({shared_count.value}/{total_count.value}) : source - {os.path.basename(source_dir)} / target - {os.path.basename(target_dir)}')

    # Check if registration file already exist.
    if version == 'icp':
        save_path = os.path.join(source_dir, os.path.basename(source_dir)) + '_' + os.path.basename(target_dir) + '_correspondence.json'
    else:
        save_path = os.path.join(source_dir, os.path.basename(source_dir)) + '_' + os.path.basename(
            target_dir) + '_correspondence_' + version + '.json'
    if os.path.exists(save_path):
        proc_log.info(log_prefix + f'Already exist file. : {save_path}')
        return

    source_mesh, source_path = util.get_mesh(source_dir)
    source_lm_dict = lm.get_lm2v(source_dir)
    target_mesh, target_path = util.get_mesh(target_dir)
    target_lm_dict = lm.get_lm2v(target_dir)

    if not source_lm_dict or not target_lm_dict:
        return

    # source & target segmentation
    start = time.time()
    with ThreadPool(processes=2) as p:
        if version == 'icp':
            result = p.starmap(segmentation, [[source_mesh, source_lm_dict], [target_mesh, target_lm_dict]])
        else:
            result = p.starmap(segmentation, [[source_mesh, source_lm_dict, 7], [target_mesh, target_lm_dict, 7]])
        source_segment = result[0]
        target_segment = result[1]
    end = time.time()
    # proc_log.info(log_prefix + f'total segmentation time using multiprocessing: {end - start}')

    icp_merge_vertices = [[0, 0, 0]] * len(source_mesh.vertices)
    nricp_merge_vertices = [[0, 0, 0]] * len(source_mesh.vertices)

    pool_input = [[source_mesh, target_mesh, ss, ts] for ss, ts in zip(source_segment, target_segment)]

    # icp
    if version == 'icp':
        start = time.time()
        with ThreadPool(processes=6) as p:
            for result in p.starmap(seg_icp, pool_input):
                s_m_vidx, icp_vertices = result
                for i, v_idx in enumerate(s_m_vidx):
                    icp_merge_vertices[v_idx] = icp_vertices[i]
        end = time.time()
        # proc_log.info(log_prefix + f'total icp time using multiprocessing: {end-start}')

        icp_source_mesh = source_mesh.copy()
        icp_source_mesh.vertices = icp_merge_vertices
        icp_source_mesh = trimesh.smoothing.filter_laplacian(icp_source_mesh, iterations=50, volume_constraint=False)
        save_mesh = icp_source_mesh

    # nricp
    else:
        start = time.time()
        with ThreadPool(processes=7) as p:
            for result in p.starmap(seg_nricp, pool_input):
                s_m_vidx, icp_vertices = result
                for i, v_idx in enumerate(s_m_vidx):
                    nricp_merge_vertices[v_idx] = icp_vertices[i]
        end = time.time()
        # proc_log.info(log_prefix + f'total nricp time using multiprocessing: {end - start}')
        nricp_source_mesh = source_mesh.copy()
        nricp_source_mesh.vertices = nricp_merge_vertices
        save_mesh = nricp_source_mesh

    query = trimesh.proximity.ProximityQuery(target_mesh)
    closest_points = query.vertex(save_mesh.vertices)[1]
    data = dict()
    data['correspondence'] = []
    for i in range(len(save_mesh.vertices)):
        new_dict = dict()
        new_dict['source_index'] = i
        new_dict['source_icp_vertex'] = list(save_mesh.vertices[i])
        new_dict['target_index'] = closest_points[i]
        new_dict['target_vertex'] = list(target_mesh.vertices[closest_points[i]])
        data['correspondence'].append(new_dict)
    with open(save_path, 'w') as out:
        json.dump(data, out, cls=NumpyEncoder, indent='\t')

    # visualize registration result
    if visual:
        p = pv.Plotter(shape=(1, 2), window_size=[800 * 2, 768])
        p.set_background('#FFFAF0')

        # get pv mesh
        raw_source_pv = pv.read(source_path)
        target_pv = pv.read(target_path)

        # visualize raw mesh
        p.subplot(0, 0)
        p.add_mesh(raw_source_pv, label='source raw', opacity=0.75, color='red')
        p.add_mesh(target_pv, label='target', opacity=0.5, color='gray')
        p.add_legend(bcolor=None, face=None, loc='lower right', size=(0.2, 0.2))
        p.camera_position = 'xy'

        # visualize result mesh
        p.subplot(0, 1)
        faces = save_mesh.faces
        count = np.array([3] * len(faces)).reshape(len(faces), 1)
        source_faces = np.concatenate((count, faces), axis=1)
        icp_source_pv = pv.PolyData(np.array(save_mesh.vertices), source_faces)
        label = 'source icp' if version == 'icp' else 'source nricp'
        p.add_mesh(icp_source_pv, label=label, opacity=0.75, color='red')
        p.add_mesh(target_pv, label='target', opacity=0.5, color='gray')
        p.add_legend(bcolor=None, face=None, loc='lower right', size=(0.2, 0.2))
        p.camera_position = 'xy'

        p.show()

