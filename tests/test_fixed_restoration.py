import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomaker.utils.restoration_contract import fixed_restoration_spec
from biomaker.utils.restoration_evaluator_template import _FIXED_RESTORATION_EVALUATOR

PROTOCOL = dict(input_column='noisy_image_path', target_column='clean_image_path',
                data_range=255, crop_size=8, group_column='noise_level',
                groups=['raw','avg2'], aggregation='mean_groups')


def task(protocol=PROTOCOL):
    return 'Primary evaluation metric: PSNR.\nSecondary evaluation metric: SSIM.\n```restoration-evaluation\n'+json.dumps(protocol)+'\n```'


def context():
    df = pd.DataFrame({'id':['01','02'], 'noisy_image_path':['a.png','b.png'], 'noise_level':['raw','avg2']})
    return dict(split_manifest={'key_columns':['id']},score_df=df, test_df=df.copy(),
                hidden_df=pd.DataFrame({'id':['01','02'],'clean_image_path':['c.png','d.png']}),
                answer_columns=['clean_image_path'])


def module(spec=None):
    result = {}
    exec(_FIXED_RESTORATION_EVALUATOR.replace('__BIOMAKER_SPEC__', repr(spec or fixed_restoration_spec(context(),task()))),result)
    return result


@pytest.mark.parametrize('constant',[False,True])
def test_independent_reference(constant):
    reference = pytest.importorskip("skimage.metrics")
    structural_similarity = reference.structural_similarity
    peak_signal_noise_ratio = reference.peak_signal_noise_ratio
    rng=np.random.default_rng(3)
    a=rng.uniform(0,255,(32,32)) if not constant else np.full((32,32),70.)
    b=rng.uniform(0,255,(32,32)) if not constant else a+3
    m=module()['_metrics'](a,b)
    assert m['psnr']==pytest.approx(peak_signal_noise_ratio(a,b,data_range=255),abs=1e-12)
    assert m['ssim']==pytest.approx(structural_similarity(a,b,data_range=255,win_size=7,gaussian_weights=False,use_sample_covariance=True),abs=1e-12)


def fixtures(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows=[];truth=[];pred=[]
    # Unequal group sizes distinguish mean_groups from a pooled image mean.
    for i,(group,error) in enumerate([('raw',1),('raw',3),('avg2',20)]):
        a=np.arange(256,dtype=np.uint8).reshape(16,16)//2
        Image.fromarray(a).save(f't{i}.png')
        b=a.astype(np.float32)+error
        Image.fromarray(b).save(f'p{i}.tif')
        rows.append(dict(id=f'{i:02}',noisy_image_path=f't{i}.png',noise_level=group))
        truth.append(dict(id=f'{i:02}',clean_image_path=f't{i}.png'))
        pred.append(dict(id=f'{i:02}',restored_image_path=f'p{i}.tif'))
    pd.DataFrame(rows).to_csv('expected.csv',index=False)
    pd.DataFrame(truth[::-1]).to_csv('truth.csv',index=False)
    pd.DataFrame(pred[::-1]).to_csv('pred.csv',index=False)
    return module()


def test_group_aggregation_alignment_and_shared_splits(tmp_path,monkeypatch):
    m=fixtures(tmp_path,monkeypatch)
    result=m['score']('pred.csv','truth.csv','expected.csv')
    raw=(20*np.log10(255)+20*np.log10(255/3))/2
    avg=20*np.log10(255/20)
    assert result['metrics']['psnr']==pytest.approx((raw+avg)/2)
    assert result['metrics']['psnr/raw']==pytest.approx(raw)
    assert result['metrics']['psnr/avg2']==pytest.approx(avg)
    m['SPEC']['score_split']='expected.csv';m['SPEC']['validation_artifact']='pred.csv'
    monkeypatch.setenv('BIOMAKER_SCORE_VAL_PATH',str(tmp_path/'truth.csv'))
    assert m['score_val']()==result
    # Private test uses different row IDs/counts and explicit paths, not validation metadata.
    e=pd.read_csv('expected.csv',dtype=str).iloc[[0,2]]
    e.to_csv('test.csv',index=False)
    for source,dest in [('pred.csv','test_pred.csv'),('truth.csv','test_truth.csv')]:
        d=pd.read_csv(source,dtype=str);d[d.id.isin(e.id)].to_csv(dest,index=False)
    assert m['score']('test_pred.csv','test_truth.csv','test.csv')['metrics']['psnr']==pytest.approx((20*np.log10(255)+avg)/2)


def test_corner_mean_not_whole_image_psnr():
    m=module();a=np.zeros((16,16));b=np.ones((16,16));b[8:,8:]=10
    score=np.mean([m['_metrics'](x,y)['psnr'] for x,y in zip(m['_crops'](a),m['_crops'](b))])
    assert score != pytest.approx(m['_metrics'](a,b)['psnr'])


@pytest.mark.parametrize('kind',['duplicate','wrong_key','missing_group','shape','nan','rgb'])
def test_invalid_artifacts_rejected(tmp_path,monkeypatch,kind):
    m=fixtures(tmp_path,monkeypatch)
    if kind in ('duplicate','wrong_key'):
        d=pd.read_csv('pred.csv',dtype=str);d.loc[0,'id']=d.loc[1,'id'] if kind=='duplicate' else 'unknown';d.to_csv('pred.csv',index=False)
    elif kind=='missing_group':
        d=pd.read_csv('expected.csv');d['noise_level']='raw';d.to_csv('expected.csv',index=False)
    elif kind=='shape':Image.fromarray(np.zeros((8,8),dtype=np.float32)).save('p0.tif')
    elif kind=='nan':Image.fromarray(np.full((16,16),np.nan,dtype=np.float32)).save('p0.tif')
    else:Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save('p0.tif')
    with pytest.raises(ValueError):m['score']('pred.csv','truth.csv','expected.csv')


def test_zero_mse_and_prediction_clipping(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='infinite'):m['_metrics'](np.zeros((8,8)),np.zeros((8,8)))
    p=tmp_path/'outside.tif';Image.fromarray(np.full((8,8),300,dtype=np.float32)).save(p)
    assert np.all(m['_image'](p,True)==255)
    with pytest.raises(ValueError,match='range'):m['_image'](p)


@pytest.mark.parametrize('change',[{'crop_size':3},{'data_range':float('nan')},{'groups':['raw','raw']},{'aggregation':'median'},{'unknown':1}])
def test_invalid_protocol(change):
    p={**PROTOCOL,**change}
    with pytest.raises(ValueError):fixed_restoration_spec(context(),task(p))


def test_admission_and_fmd_contract():
    assert fixed_restoration_spec(context(),'Primary evaluation metric: PSNR.') is None
    c=context();c['test_df']=c['test_df'].drop(columns=['noise_level'])
    with pytest.raises(ValueError):fixed_restoration_spec(c,task())
    text=task({**PROTOCOL,'crop_size':256,'groups':['raw','avg2','avg4','avg8','avg16']});c=context()
    for k in ('score_df','test_df'):
        c[k]=pd.DataFrame({'id':list('abcde'),'noisy_image_path':['x.png']*5,'noise_level':['raw','avg2','avg4','avg8','avg16']})
    s=fixed_restoration_spec(c,text)
    assert s['restoration_evaluation']['crop_size']==256
    assert s['metrics']==[{'name':'psnr','maximize':True},{'name':'ssim','maximize':True}]


def test_frozen_writer_and_format_only_submission(tmp_path,monkeypatch):
    from biomaker.utils.evaluator_setup import _write_fixed_evaluator
    m=fixtures(tmp_path,monkeypatch)
    split=tmp_path/'input/biomaker_splits';split.mkdir(parents=True)
    (split/'agent_test.csv').write_text(Path('expected.csv').read_text())
    spec=copy.deepcopy(m['SPEC']);spec['score_split']='expected.csv'
    prepared=_write_fixed_evaluator(tmp_path,tmp_path/'logs',tmp_path/'truth.csv',60,spec,
                                   _FIXED_RESTORATION_EVALUATOR,run_public_contract=False,log_message='test')
    assert prepared is not None
    assert json.loads(prepared.prediction_spec)['restoration_evaluation']==PROTOCOL
    # Format validation never needs hidden truth.
    Path('working').mkdir(exist_ok=True)
    Path('working/submission.csv').write_text(Path('pred.csv').read_text())
    Path('truth.csv').unlink()
    assert m['validate_submission']()['valid']


def test_whole_image_and_image_weighting(tmp_path, monkeypatch):
    m = fixtures(tmp_path, monkeypatch)
    m['P']['crop_size'] = None
    m['P']['aggregation'] = 'mean_images'
    got = m['score']('pred.csv', 'truth.csv', 'expected.csv')
    expected = np.mean([20*np.log10(255/e) for e in (1, 3, 20)])
    assert got['metrics']['psnr'] == pytest.approx(expected)
    m['P']['group_column'] = None
    m['P']['groups'] = None
    ungrouped = m['score']('pred.csv', 'truth.csv', 'expected.csv')
    assert ungrouped['metrics']['psnr'] == pytest.approx(expected)
    assert ungrouped['per_group'] == {}


@pytest.mark.parametrize('keys', [['noisy_image_path'], ['noisy_image_path', 'noise_level']])
def test_input_path_can_be_sample_key(tmp_path, monkeypatch, keys):
    original = fixtures(tmp_path, monkeypatch)
    baseline = original['score']('pred.csv', 'truth.csv', 'expected.csv')
    expected = pd.read_csv('expected.csv', dtype=str)
    for name in ['pred.csv', 'truth.csv']:
        frame = pd.read_csv(name, dtype=str)
        expected.merge(frame, on='id').drop(columns='id').to_csv(name, index=False)
    expected = expected.drop(columns='id')
    expected.to_csv('expected.csv', index=False)
    ctx = dict(split_manifest={'key_columns': keys}, score_df=expected,
               test_df=expected.copy(), hidden_df=pd.read_csv('truth.csv'),
               answer_columns=['clean_image_path'])
    spec = fixed_restoration_spec(ctx, task())
    scorer = module(spec)
    assert scorer['score']('pred.csv', 'truth.csv', 'expected.csv') == baseline
    splits = Path('input/biomaker_splits')
    splits.mkdir(parents=True)
    expected.to_csv(splits/'agent_test.csv', index=False)
    scorer['SPEC']['score_split'] = 'expected.csv'
    monkeypatch.setenv('BIOMAKER_SCORE_VAL_PATH', str(tmp_path/'truth.csv'))
    scorer['self_test']()
    assert scorer['validate_submission']()['valid']


@pytest.mark.parametrize('key', ['clean_image_path', 'restored_image_path'])
def test_hidden_or_output_key_rejected(key):
    ctx = context()
    ctx['split_manifest']['key_columns'] = [key]
    with pytest.raises(ValueError, match='cannot be sample keys'):
        fixed_restoration_spec(ctx, task())


def test_shared_input_requires_unique_composite_identity():
    ctx = context()
    for name in ('score_df', 'test_df'):
        ctx[name]['noisy_image_path'] = 'shared.png'
    ctx['split_manifest']['key_columns'] = ['noisy_image_path', 'noise_level']
    assert fixed_restoration_spec(ctx, task())['key_columns'] == ['noisy_image_path', 'noise_level']
    ctx['split_manifest']['key_columns'] = ['noisy_image_path']
    with pytest.raises(ValueError, match='unique'):
        fixed_restoration_spec(ctx, task())


@pytest.mark.parametrize('keys', ['id', ['id','id'], [''], [None]])
def test_invalid_key_declarations(keys):
    ctx = context()
    ctx['split_manifest']['key_columns'] = keys
    with pytest.raises(ValueError, match='sample keys'):
        fixed_restoration_spec(ctx, task())


def test_agent_schema_without_task_json(tmp_path, monkeypatch):
    baseline_module = fixtures(tmp_path, monkeypatch)
    baseline = baseline_module['score']('pred.csv', 'truth.csv', 'expected.csv')
    rename = dict(id='sample', noisy_image_path='source', clean_image_path='reference',
                  noise_level='condition', restored_image_path='estimate')
    for name in ['expected.csv','truth.csv','pred.csv']:
        df = pd.read_csv(name, dtype=str).rename(columns=rename)
        df.to_csv(name,index=False)
    c = dict(split_manifest={'key_columns':['sample'],
                'metric_contract': {**PROTOCOL, 'input_column':'source',
                                    'target_column':'reference','group_column':'condition'},
                'prediction_contract': {'prediction_columns':['estimate']}},
             score_df=pd.read_csv('expected.csv'),test_df=pd.read_csv('expected.csv'),
             hidden_df=pd.read_csv('truth.csv'),answer_columns=['reference'])
    spec = fixed_restoration_spec(c, 'Primary evaluation metric: PSNR.\nSecondary evaluation metric: SSIM.')
    assert spec['prediction_columns']==['estimate']
    assert module(spec)['score']('pred.csv','truth.csv','expected.csv')==baseline
    c['split_manifest']['prediction_contract']['prediction_columns']=['ambiguous','two']
    assert fixed_restoration_spec(c, 'Primary evaluation metric: PSNR.') is None


@pytest.mark.parametrize('settings', [
    {'crop_mode': 'center'},
    {'crop_mode': 'whole_image'},
    {'ssim_parameters': {'unknown_option': 7}},
    {'ssim_parameters': {'win_size': 7, 'window_size': 9}},
    {'undeclared_weighting': 'pixels'},
])
def test_unsupported_manifest_protocol_is_explicit_error(settings):
    c=context()
    c['split_manifest']['metric_contract']={**PROTOCOL, **settings}
    with pytest.raises(ValueError):
        fixed_restoration_spec(c, 'Primary evaluation metric: PSNR.')


@pytest.mark.parametrize('in_task', [False, True])
def test_window_alias_and_crop_mode_are_frozen(in_task):
    protocol = {**PROTOCOL, 'crop_mode': 'four_corners',
                'ssim_parameters': {'window_size': 7}}
    c = context()
    if in_task:
        text = task(protocol)
    else:
        c['split_manifest']['metric_contract'] = protocol
        text = 'Primary evaluation metric: PSNR.\nSecondary evaluation metric: SSIM.'
    spec = fixed_restoration_spec(c, text)
    assert spec['ssim_parameters']['win_size'] == 7
    assert 'window_size' not in spec['ssim_parameters']
    assert spec['restoration_evaluation']['ssim_parameters'] == spec['ssim_parameters']
    assert spec['restoration_evaluation']['crop_mode'] == 'four_corner_crops_no_center'
    rng = np.random.default_rng(17)
    truth, prediction = rng.uniform(0, 255, (2, 16, 16))
    assert module(spec)['_metrics'](truth, prediction) == module()['_metrics'](truth, prediction)


def test_nested_and_outer_ssim_conflict_is_rejected():
    c = context()
    c['split_manifest']['metric_contract'] = {
        'restoration_evaluation': {**PROTOCOL, 'ssim_parameters': {'win_size': 7}},
        'ssim_parameters': {'window_size': 9},
    }
    with pytest.raises(ValueError, match='Conflicting'):
        fixed_restoration_spec(c, 'Primary evaluation metric: PSNR.')


@pytest.mark.parametrize('parameters', [
    {'window_size': 5, 'use_sample_covariance': False, 'K1': .02},
    {'gaussian_weights': True},
])
def test_frozen_template_matches_shared_metric_kernel(parameters):
    from biomaker.utils.trusted_metrics import restoration_metrics

    spec = fixed_restoration_spec(context(), task({
        **PROTOCOL, 'crop_size': 16, 'ssim_parameters': parameters,
    }))
    rng = np.random.default_rng(13)
    truth, prediction = rng.uniform(0, 255, (2, 32, 32))
    trusted = restoration_metrics(truth, prediction, 255, parameters)
    assert module(spec)['_metrics'](truth, prediction) == trusted


def test_psnr_only_does_not_impose_ssim_window():
    protocol = {**PROTOCOL, 'crop_size': 2}
    text = 'Primary evaluation metric: PSNR.\n```restoration-evaluation\n' + json.dumps(protocol) + '\n```'
    spec = fixed_restoration_spec(context(), text)
    evaluator = module(spec)
    truth, prediction = np.zeros((2, 2)), np.ones((2, 2))
    assert len(evaluator['_crops'](truth)) == 4
    assert evaluator['_metrics'](truth, prediction) == {'psnr': pytest.approx(20*np.log10(255))}


def test_ssim_only_allows_perfect_prediction():
    text = 'Primary evaluation metric: SSIM.\n```restoration-evaluation\n' + json.dumps(PROTOCOL) + '\n```'
    spec = fixed_restoration_spec(context(), text)
    truth = np.zeros((8, 8))
    assert module(spec)['_metrics'](truth, truth) == {'ssim': pytest.approx(1)}


def test_numpy_prediction_matches_image_prediction(tmp_path,monkeypatch):
    m=fixtures(tmp_path,monkeypatch)
    expected=m['score']('pred.csv','truth.csv','expected.csv')
    frame=pd.read_csv('pred.csv',dtype=str)
    for i,path in enumerate(frame['restored_image_path']):
        np.save(f'prediction{i}.npy',np.asarray(Image.open(path)))
        frame.loc[i,'restored_image_path']=f'prediction{i}.npy'
    frame.to_csv('pred.csv',index=False)
    assert m['score']('pred.csv','truth.csv','expected.csv')==expected
    np.save('bad.npy',np.ones((16,16),dtype=complex))
    with pytest.raises(ValueError,match='real numeric'):m['_image']('bad.npy',True)
