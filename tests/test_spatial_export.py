"""Offline export integration on synthetic arrays, including native class order."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from flakepipeline.spatial_maps import export_spatial_maps, _load_sources
from flakepipeline.spatial_maps.stats import sha


@pytest.fixture
def sources(tmp_path):
    native = np.array([[0,1,2,2,3,3],[1,2,2,2,0,3],[3,2,1,2,3,3],[0,2,2,2,1,3]],np.uint8)
    valid = np.ones(native.shape,bool); valid[0,0]=False;valid[-1,-1]=False
    excluded = np.zeros(native.shape,bool);excluded[1,1]=True;excluded[0,0]=True
    paths=[]
    for name,arr in [('mask',native),('valid',valid),('excluded',excluded)]:
        p=tmp_path/(name+'.npy');np.save(p,arr);paths.append(p)
    return paths,native,valid,excluded


def test_native_and_colour_exports_agree_with_exact_counts(sources,tmp_path):
    paths,native,valid,excluded=sources
    original=[sha(p) for p in paths]
    export=export_spatial_maps(*paths,tmp_path/'native','native_sample',(3,2),(2,2))
    manifest=json.loads(Path(export['manifest']).read_text())
    # Source convention: native value 1 is bilayer, 2 monolayer.
    kept=native[valid & ~excluded]
    counts={'background':int((kept==0).sum()),'1L':int((kept==2).sum()),
            '2L':int((kept==1).sum()),'TL':int((kept==3).sum())}
    assert manifest['global']['counts_px']==counts
    assert manifest['global']['excluded_px']==1
    assert manifest['global']['non_scan_px']==2
    assert len(export['outputs'])==16
    for record in manifest['outputs']:
        p=Path(export['manifest']).parent/record['path']
        assert sha(p)==record['sha256']
    with (tmp_path/'native/native_sample_heatmap_tiles.csv').open() as f: rows=list(csv.DictReader(f))
    for layer in ('1L','2L','TL'):assert sum(int(r[layer+'_px']) for r in rows)==counts[layer]
    with np.load(tmp_path/'native/native_sample_heatmaps_arrays.npz') as data:
        assert data['1L_px'].shape==(2,3)
        assert int(data['1L_px'].sum())==counts['1L']
    palette=np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0]],np.uint8)
    png=tmp_path/'mask.png';Image.fromarray(palette[native]).save(png)
    color=export_spatial_maps(png,paths[1],paths[2],tmp_path/'colour','colour_sample',(3,2),(2,2))
    assert color['summary']==export['summary']
    assert [sha(p) for p in paths]==original


def test_render_failure_does_not_publish_partial_run(sources,tmp_path,monkeypatch):
    from flakepipeline.spatial_maps import heatmaps
    def fail(*args):raise RuntimeError('synthetic renderer failure')
    monkeypatch.setattr(heatmaps,'plot_sample',fail)
    with pytest.raises(RuntimeError,match='synthetic'):
        export_spatial_maps(*sources[0],tmp_path/'out','S',(2,2),(2,2))
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('.spatial-staging-*'))


def test_existing_output_preserved(sources,tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'prior.txt').write_text('keep')
    with pytest.raises(ValueError,match='already exists'):
        export_spatial_maps(*sources[0],out,'S',(2,2),(2,2))
    assert (out/'prior.txt').read_text()=='keep'


@pytest.mark.parametrize('name',['../bad','a/b','a\\b','a:bad','..','', 'name.'])
def test_unsafe_sample_names_rejected(sources,tmp_path,name):
    with pytest.raises(ValueError,match='safe filename'):
        export_spatial_maps(*sources[0],tmp_path/'out',name,(2,2),(2,2))
    assert not (tmp_path/'out').exists()


def test_native_invalid_ids_rejected(tmp_path):
    p=tmp_path/'mask.npy';np.save(p,np.array([[0,4]],np.uint8))
    with pytest.raises(ValueError,match='Native mask'):_load_sources(p,None,None)


def test_heatmap_marks_partial_npy_support(sources):
    from flakepipeline.spatial_maps.stats import analyze
    from flakepipeline.spatial_maps import heatmaps
    from matplotlib import pyplot as plt
    paths,_,valid,excluded=sources
    labels,*_= _load_sources(*paths)
    table,xs,ys,whole=analyze(labels,[],3,2,valid,excluded)
    result=dict(table=table,xedges=xs,yedges=ys,rectangles=[],width=6,height=4)
    fig,ax=plt.subplots()
    heatmaps.heatmap(ax,result,'1L_of_crystal_pct','1L',True)
    assert any('†' in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_empty_crystal_cli_exports_null_fractions(tmp_path):
    p=tmp_path/'blank.npy';np.save(p,np.zeros((4,6),np.uint8))
    root=Path(__file__).resolve().parents[1]
    run=subprocess.run([sys.executable,str(root/'tools/layer_spatial_maps.py'),'--mask',str(p),
        '--sample','blank','--output',str(tmp_path/'out'),'--heatmap-grid','3','2','--combined-grid','3','2'],
        capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    result=json.loads((tmp_path/'out/spatial_manifest.json').read_text())
    assert result['global']['of_crystal_pct']=={'1L':None,'2L':None,'TL':None}
    assert result['global']['crystal_coverage_pct']==0
