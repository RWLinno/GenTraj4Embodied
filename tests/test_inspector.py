import unittest
import tempfile
from pathlib import Path
import h5py
import numpy as np
from inspector.audit import inspect_arrays, audit

class InspectorTests(unittest.TestCase):
    def arrays(self):
        p=np.zeros((3,7));p[:,0]=[0,1,3];p[:,6]=1
        return p,np.array([0.,.5,1.5]),p[0].copy(),p[-1].copy()
    def test_nonuniform_time_constant_velocity(self):
        r=inspect_arrays(*self.arrays());self.assertEqual(r['max_speed'],2);self.assertEqual(r['max_acceleration'],0);self.assertFalse(r['issues'])
    def test_quaternion_sign_is_equivalent(self):
        p,t,start,end=self.arrays();end[3:]*=-1
        self.assertNotIn('orientation_endpoint_mismatch',inspect_arrays(p,t,start,end)['issues'])
    def test_duplicate_timestamp_rejects_derivatives(self):
        p,t,s,e=self.arrays();t[1]=0;r=inspect_arrays(p,t,s,e)
        self.assertIn('nonincreasing_timestamps',r['issues']);self.assertNotIn('max_speed',r)
    def test_bad_shape_and_nonfinite(self):
        p,t,s,e=self.arrays();self.assertIn('invalid_shape',inspect_arrays(p[:,:3],t,s,e)['issues'])
        p[1,0]=np.nan;self.assertIn('nonfinite_values',inspect_arrays(p,t,s,e)['issues'])
    def test_norm_and_endpoint_flags(self):
        p,t,s,e=self.arrays();p[1,6]=0;e[0]+=1
        r=inspect_arrays(p,t,s,e);self.assertIn('nonunit_quaternion',r['issues']);self.assertIn('position_endpoint_mismatch',r['issues'])
    def test_cross_split_duplicates_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=[]
            for name in ['train.h5','test.h5']:
                path=Path(tmp)/name;files.append(path)
                with h5py.File(path,'w') as f:
                    g=f.create_group('trajectory_0')
                    for key,value in zip(['trajectory','timestamps','start_pose','end_pose'],self.arrays()):g[key]=value
            r=audit(files+[Path(tmp)/'missing.h5'])
            self.assertEqual(len(r['cross_file_exact_path_duplicates']),1)
            self.assertEqual(len(r['cross_file_endpoint_overlaps']),1)
            self.assertEqual(r['sources'][-1]['issues'],['missing_input'])
    def test_corrupt_record_does_not_hide_remaining_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.h5'
            with h5py.File(path,'w') as f:f.create_group('trajectory_0')
            r=audit([path]);self.assertEqual(r['summary']['issues']['unreadable_record'],1)

    def test_cli_strict_and_safe_html(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unsafe.h5'; output=Path(tmp)/'report'
            with h5py.File(path,'w') as f:
                g=f.create_group('trajectory_0')
                for key,value in zip(['trajectory','timestamps','start_pose','end_pose'],self.arrays()):g[key]=value
                g.attrs['task_name']='</script><script>alert(1)</script>'
                g.attrs['length']=999
            result=subprocess.run([sys.executable,'-m','inspector.audit',str(path),'--out',str(output),'--strict'],capture_output=True)
            self.assertEqual(result.returncode,1)
            html=(output/'index.html').read_text()
            self.assertNotIn('</script><script>alert(1)</script>',html)
            self.assertIn('length_metadata_mismatch',(output/'report.json').read_text())


class DiagnosticTests(unittest.TestCase):
    def test_histograms_keep_outliers_and_account_for_missing(self):
        from inspector.audit import diagnostic_data
        report={'sources':[{'file':'a'}], 'records':[{'source':'a','task':'x','max_speed':v} for v in [0,1,2,100]]+[{'source':'a','task':'y'}]}
        d=diagnostic_data(report);h=d['histograms']['max_speed']
        self.assertEqual(sum(h['counts']),4)
        self.assertEqual(h['excluded'],1)
        self.assertEqual(h['edges'][-1],100)
        self.assertEqual(h['quantiles']['p50'],1.5)
        self.assertEqual(d['splits'][0]['tasks'],{'x':4,'y':1})
        self.assertIsNone(d['histograms']['max_acceleration'])

if __name__=='__main__':unittest.main()
