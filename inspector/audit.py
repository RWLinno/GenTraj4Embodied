"""Audit group-per-trajectory HDF5 files without importing any model code."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter, defaultdict
import h5py
import numpy as np

VERSION = '1.0.0'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_pose(pose):
    p = np.array(pose, dtype=float, copy=True)
    q = p[3:]
    norm = np.linalg.norm(q)
    if norm:
        q /= norm
        # q and -q encode the same orientation, in either common component order.
        pivot = np.argmax(np.abs(q))
        if q[pivot] < 0:
            q *= -1
    return np.round(p, 6).tolist()


def rotation_error(q, target):
    norms = np.linalg.norm(q) * np.linalg.norm(target)
    if norms <= 1e-15:
        return None
    return float(2 * np.arccos(np.clip(abs(np.dot(q, target) / norms), 0, 1)))


def inspect_arrays(trajectory, timestamps, start, end):
    """Coordinates retain input units; derivatives use actual timestamps."""
    p, t = np.asarray(trajectory, dtype=float), np.asarray(timestamps, dtype=float)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    row = {'issues': []}
    if p.ndim != 2 or p.shape[1] != 7 or len(p) < 2 or t.shape != (len(p),) or start.shape != (7,) or end.shape != (7,):
        row['issues'].append('invalid_shape')
        return row
    row['waypoints'] = len(p)
    if not all(np.isfinite(x).all() for x in [p, t, start, end]):
        row['issues'].append('nonfinite_values')
        return row
    norms = np.linalg.norm(p[:, 3:], axis=1)
    row['max_quaternion_norm_error'] = float(np.max(abs(norms - 1)))
    if row['max_quaternion_norm_error'] > 1e-3 or max(abs(np.linalg.norm(start[3:])-1), abs(np.linalg.norm(end[3:])-1)) > 1e-3:
        row['issues'].append('nonunit_quaternion')
    row['start_position_error'] = float(np.linalg.norm(p[0, :3] - start[:3]))
    row['end_position_error'] = float(np.linalg.norm(p[-1, :3] - end[:3]))
    row['start_orientation_error_rad'] = rotation_error(p[0, 3:], start[3:])
    row['end_orientation_error_rad'] = rotation_error(p[-1, 3:], end[3:])
    if max(row['start_position_error'], row['end_position_error']) > 1e-6:
        row['issues'].append('position_endpoint_mismatch')
    if any(v is None or v > 1e-6 for v in [row['start_orientation_error_rad'],row['end_orientation_error_rad']]):
        row['issues'].append('orientation_endpoint_mismatch')
    d = np.diff(p[:, :3], axis=0)
    row['path_length'] = float(np.linalg.norm(d, axis=1).sum())
    row['duration'] = float(t[-1] - t[0])
    dt = np.diff(t)
    if np.any(dt <= 0):
        row['issues'].append('nonincreasing_timestamps')
    else:
        velocity = d / dt[:, None]
        row['max_speed'] = float(np.linalg.norm(velocity, axis=1).max())
        if len(p) >= 3:
            midpoints = (t[1:] + t[:-1]) / 2
            acceleration = np.diff(velocity, axis=0) / np.diff(midpoints)[:, None]
            row['max_acceleration'] = float(np.linalg.norm(acceleration, axis=1).max())
    row['_condition'] = json.dumps([canonical_pose(start),canonical_pose(end)], separators=(',', ':'))
    # Equality of stored numerical values, not similarity of resampled paths.
    row['_trajectory_hash'] = hashlib.sha256(np.ascontiguousarray(np.column_stack([t,p]),dtype='<f8').tobytes()).hexdigest()
    return row


def audit(inputs):
    records, sources, previews = [], [], []
    for filename in inputs:
        path = Path(filename)
        source = {'file':path.name,'sha256':None,'records':0,'issues':[]}
        sources.append(source)
        if not path.is_file():
            source['issues'].append('missing_input'); continue
        source['sha256'] = sha256(path)
        try:
            with h5py.File(path, 'r') as f:
                groups = sorted(k for k in f if isinstance(f[k], h5py.Group))
                if not groups:
                    source['issues'].append('no_trajectory_groups')
                declared = f.attrs.get('num_trajectories')
                if declared is not None and int(declared) != len(groups):
                    source['issues'].append('record_count_mismatch')
                source['records'] = len(groups)
                source['declared_waypoints'] = int(f.attrs['trajectory_length']) if 'trajectory_length' in f.attrs else None
                for key in groups:
                    g = f[key]
                    row = {'source':path.name,'group':key,'issues':[]}
                    try:
                        values = [g[k][()] for k in ['trajectory','timestamps','start_pose','end_pose']]
                        row.update(inspect_arrays(*values))
                        task = g.attrs.get('task_name','unknown')
                        row['task'] = task.decode('utf-8',errors='replace') if isinstance(task,bytes) else str(task)
                        row['modality'] = str(g.attrs.get('modality','unknown'))
                        if 'length' in g.attrs and row.get('waypoints') != int(g.attrs['length']):
                            row['issues'].append('length_metadata_mismatch')
                        if len([p for p in previews if p['source']==path.name]) < 8 and '_condition' in row:
                            previews.append({'source':path.name,'group':key,'task':row['task'],'issues':row['issues'].copy(),
                                             'points':np.asarray(values[0])[:,:3].round(6).tolist()})
                    except (KeyError,ValueError,TypeError,OSError) as e:
                        row['issues'].append('unreadable_record')
                        row['error_type'] = type(e).__name__
                    records.append(row)
        except (OSError,ValueError) as e:
            source['issues'].append('unreadable_hdf5')
    issues = Counter(i for r in records for i in r['issues'])
    lengths = Counter(str(r['waypoints']) for r in records if 'waypoints' in r)
    tasks = Counter(r.get('task','unknown') for r in records)
    conditions, duplicate_paths = defaultdict(list), defaultdict(list)
    for index, r in enumerate(records):
        if '_condition' in r: conditions[r.pop('_condition')].append(index)
        if '_trajectory_hash' in r: duplicate_paths[r.pop('_trajectory_hash')].append(index)
    def overlaps(groups):
        return [{'records':[{'source':records[i]['source'],'group':records[i]['group']} for i in ids]}
                for ids in groups.values() if len({records[i]['source'] for i in ids}) > 1]
    def distribution(key):
        a = [r[key] for r in records if key in r and r[key] is not None]
        return {'min':float(np.min(a)),'median':float(np.median(a)),'max':float(np.max(a))} if a else None
    return {'version':VERSION,'sources':sources,'summary':{
        'records':len(records),'flagged_records':sum(bool(r['issues']) for r in records),
        'issues':dict(issues),'waypoint_counts':dict(sorted(lengths.items(),key=lambda x:int(x[0]))),'tasks':dict(tasks),
        'unique_endpoint_conditions':len(conditions),'repeated_endpoint_conditions':sum(len(ids)>1 for ids in conditions.values()),
        'speed':distribution('max_speed'),'acceleration':distribution('max_acceleration')},
        'cross_file_exact_path_duplicates':overlaps(duplicate_paths),'cross_file_endpoint_overlaps':overlaps(conditions),
        'records':records,'previews':previews,
        'scope':'Input files only. No collision, IK, robot success, angular dynamics, or distribution-quality test. Endpoint grouping rounds poses to 6 decimals and ignores task/scene metadata. Zero flags is not a robot-safety certificate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--out',type=Path,default=Path('audit-output'))
    parser.add_argument('--strict',action='store_true',help='Exit 1 on record/source integrity flags or exact cross-file duplicates; endpoint overlaps are informational.')
    args = parser.parse_args()
    if len({Path(f).name for f in args.files}) != len(args.files):
        parser.error('Input filenames must be unique to identify sources unambiguously.')
    output = args.out.resolve()
    if any(Path(f).resolve() in [output/'report.json',output/'index.html'] for f in args.files):
        parser.error('Output would overwrite an input file.')
    report = audit(args.files)
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    compact = {k:v for k,v in report.items() if k!='records'}
    # Prevent dataset metadata from becoming markup or closing the JSON script tag.
    payload = json.dumps(compact,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template = Path(__file__).with_name('report.html').read_text()
    (output/'index.html').write_text(template.replace('__AUDIT_DATA__',payload))
    print(json.dumps(report['summary'],indent=2))
    failed = report['summary']['flagged_records'] or any(s['issues'] for s in report['sources']) or report['cross_file_exact_path_duplicates']
    return int(bool(args.strict and failed))

if __name__ == '__main__':
    raise SystemExit(main())
