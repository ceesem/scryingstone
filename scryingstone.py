import datetime
import numpy as np
import pandas as pd
from nglui import statebuilder
from concurrent.futures import ThreadPoolExecutor
from caveclient.base import handle_response
import pytz
import tqdm 

def lookup_user_info(user_id, client):
    endpoint = f"{client.server_address}/auth/api/v1/user/{user_id}"
    r = client.info.session.get(endpoint)
    return handle_response(r)

def extract_timestamp(op):
    return datetime.datetime.fromisoformat(op['timestamp'])

def extract_user(op):
    return op['user']

def extract_type(op):
    if 'removed_edges' in op:
        return 'split'
    else:
        return 'merge'
    
def extract_location(op, data_resolution, viewer_resolution):
    coords = np.vstack((op['sink_coords'], op['source_coords'])) 
    return (np.mean(coords, axis=0) * data_resolution / viewer_resolution)

def extract_point_count(op):
    coords = np.vstack((op['sink_coords'], op['source_coords']))
    return len(coords)

def extract_roots(op):
    return op['roots']


def part_of_root_id(obj_id, root_id, client):
    lg = client.chunkedgraph.get_lineage_graph(obj_id)
    targets = np.array([x['target'] for x in lg['links']])
    sources = np.array([x['source'] for x in lg['links']])
    return root_id in targets[~np.isin(targets, sources)]

def rids_not_part(row, client):
    if row["type"] == "merge":
        return []
    roots = row["roots"]
    keep_roots = []
    for root in roots:
        if not part_of_root_id(root, row["root_id"], client):
            keep_roots.append(root)
            break
    return keep_roots


def run_disjoint_roots(df, client, n_threads=20):
    exe = ThreadPoolExecutor(n_threads)
    threads = []
    for _, row in df.iterrows():
        threads.append(exe.submit(rids_not_part, row, client))
    keep_roots = [t.result() for t in threads]

    return keep_roots

def get_operations(root_id, client, id_per_chunk = 25):
    cl = client.chunkedgraph.get_change_log(root_id)
    operation_ids = np.array(cl['operations_ids'])
    if len(operation_ids) == 0:
        return {}
    elif len(operation_ids) > id_per_chunk:
        num_chunks = np.ceil( len(operation_ids) / id_per_chunk ).astype(int)
        opids_ch = np.array_split(operation_ids, num_chunks)
    else:
        opids_ch = [operation_ids]
        
    op_info = {}
    for op_ids in opids_ch:
        opi = client.chunkedgraph.get_operation_details(op_ids)
        op_info.update(opi)
    return op_info    
    
def assemble_dataframe(root_id, operation_info, data_resolution, viewer_resolution):
    obj_df = pd.DataFrame(
        {
            "root_id": root_id,
            "op_id": list(operation_info.keys()),
            'user': [int(extract_user(op)) for _, op in operation_info.items()],
            'timestamp': [extract_timestamp(op) for _, op in operation_info.items()],
            'type': [extract_type(op) for _, op in operation_info.items()],
            'location': [extract_location(op, data_resolution, viewer_resolution) for _, op in operation_info.items()],
            'n_points': [extract_point_count(op) for _, op in operation_info.items()],
            'roots': [extract_roots(op) for _, op in operation_info.items()],
        }
    )
    return obj_df
    
def edit_statebuilder(client, viewer_resolution):
    img, seg = statebuilder.from_client(client)
    seg.add_selection_map(selected_ids_column='root_id')
    pt = statebuilder.PointMapper("location", description_column='op_id', linked_segmentation_column='disjoint_roots')
    anno_merge = statebuilder.AnnotationLayerConfig(
        "merge", mapping_rules=pt, filter_query="type=='merge'", color="#2796a8",
    )
    anno_split = statebuilder.AnnotationLayerConfig(
        "split", mapping_rules=pt, filter_query="type=='split'", color="#cc4b35", linked_segmentation_layer=seg.name,
    )
    return statebuilder.StateBuilder([img, seg, anno_merge, anno_split], client=client, resolution=viewer_resolution)
    
def last_days(n):
    now = datetime.datetime.now(tz=pytz.utc)
    return now - datetime.timedelta(days=n) 

def delta_minutes(m):
    return datetime.timedelta(minutes=m)

palette = {
    "split" : (0.800, 0.294, 0.208),
    "merge" : (0.153, 0.588, 0.659),
}