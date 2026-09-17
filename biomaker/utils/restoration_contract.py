"""Explicit admission to deterministic grayscale image-restoration scoring."""
import json
import math
import re

from .evaluator_protocol import requested_metrics
from .trusted_metrics import normalize_ssim_parameters


def fixed_restoration_spec(context, task):
    primary, secondary = requested_metrics(task)
    if primary not in {'psnr', 'ssim'} or any(m not in {'psnr', 'ssim'} for m in secondary):
        return None
    blocks = re.findall(r'```restoration-evaluation\s*\n(.*?)```', task, re.S)
    if len(blocks) > 1:
        raise ValueError('Declare exactly one restoration-evaluation contract.')
    manifest = context['split_manifest']
    output = manifest.get('prediction_contract', {})
    contract = manifest.get('metric_contract', {})
    if blocks:
        protocol = json.loads(blocks[0])
    else:
        # The preparation agent declares semantics, not prescribed column names.
        # Only compile a protocol that can be mapped without guessing settings.
        source = contract.get('restoration_evaluation', contract)
        settings = {'data_range', 'crop_size', 'group_column', 'groups', 'aggregation'}
        if not isinstance(source, dict) or not settings <= source.keys():
            return None
        semantic_fields = settings | {'input_column', 'target_column', 'crop_mode', 'ssim_parameters'}
        metadata_fields = {'primary_metric', 'secondary_metrics', 'maximize', 'task_type'}
        unknown = set(source) - semantic_fields - (metadata_fields if source is contract else set())
        if source is not contract:
            unknown |= set(contract) - metadata_fields - {'restoration_evaluation', 'ssim_parameters', 'crop_mode'}
        if unknown:
            raise ValueError(f'Unsupported restoration-evaluation parameters: {sorted(unknown)}')
        ic = source.get('input_column', output.get('input_column'))
        answers = context['answer_columns']
        tc = source.get('target_column', answers[0] if len(answers) == 1 else None)
        if not ic or not tc:
            return None
        protocol = {k: source[k] for k in settings}
        protocol.update(input_column=ic, target_column=tc)
        for key in ('crop_mode', 'ssim_parameters'):
            if key in source:
                protocol[key] = source[key]
        if source is not contract and 'crop_mode' in contract:
            outer = contract['crop_mode']
            inner = protocol.get('crop_mode', outer)
            outer = 'four_corner_crops_no_center' if outer == 'four_corners' else outer
            inner = 'four_corner_crops_no_center' if inner == 'four_corners' else inner
            if inner != outer:
                raise ValueError('Conflicting outer and nested crop_mode.')
            protocol['crop_mode'] = outer
        if source is not contract and 'ssim_parameters' in contract:
            outer = normalize_ssim_parameters(contract['ssim_parameters'])
            if 'ssim_parameters' in protocol and outer != normalize_ssim_parameters(protocol['ssim_parameters']):
                raise ValueError('Conflicting outer and nested ssim_parameters.')
            protocol['ssim_parameters'] = outer
    required = {'input_column', 'target_column', 'data_range', 'crop_size',
                'group_column', 'groups', 'aggregation'}
    optional = {'crop_mode', 'ssim_parameters'}
    if not isinstance(protocol, dict) or not required <= protocol.keys() or set(protocol) - required - optional:
        raise ValueError('Incomplete or unknown restoration-evaluation parameters.')
    protocol = dict(protocol)
    ssim = normalize_ssim_parameters(protocol.get('ssim_parameters'))
    if 'ssim_parameters' in protocol:
        protocol['ssim_parameters'] = ssim
    dr = protocol['data_range']
    if isinstance(dr, bool) or not isinstance(dr, (float, int)) or not math.isfinite(dr) or dr <= 0:
        raise ValueError('data_range must be finite and positive.')
    crop = protocol['crop_size']
    minimum = ssim['win_size'] if 'ssim' in [primary, *secondary] else 1
    if crop is not None and (type(crop) is not int or crop < minimum):
        raise ValueError(f'crop_size must be null (whole image) or an integer >= {minimum}.')
    if 'crop_mode' in protocol:
        mode = protocol['crop_mode']
        if mode == 'four_corners':
            mode = 'four_corner_crops_no_center'
        if not isinstance(mode, str) or mode not in {'four_corner_crops_no_center', 'whole_image'}:
            raise ValueError(f'Unsupported restoration crop_mode: {mode!r}.')
        if (mode == 'whole_image') != (crop is None):
            raise ValueError('crop_mode and crop_size disagree: whole_image requires null; four corners require a crop size.')
        protocol['crop_mode'] = mode
    group, groups = protocol['group_column'], protocol['groups']
    if (group is None) != (groups is None):
        raise ValueError('group_column and groups must both be null or declared.')
    if group is not None and (not isinstance(group, str) or not group or
            not isinstance(groups, list) or not groups or
            any(not isinstance(g, str) or not g for g in groups) or len(set(groups)) != len(groups)):
        raise ValueError('Grouping requires a column and unique string group values.')
    if not isinstance(protocol['aggregation'], str) or protocol['aggregation'] not in {'mean_images', 'mean_groups'} or (group is None and protocol['aggregation'] == 'mean_groups'):
        raise ValueError('Unsupported aggregation.')
    keys = context['split_manifest'].get('key_columns', [])
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('Restoration scoring requires explicit sample keys.')
    ic, tc = protocol['input_column'], protocol['target_column']
    # An observed input path is a valid sample identifier, including as part
    # of a composite key. Hidden answers and predicted outputs cannot be keys.
    if not isinstance(ic, str) or not ic or not isinstance(tc, str) or not tc or ic == tc:
        raise ValueError('Declare distinct nonempty input and target path columns.')
    declared = output.get('prediction_columns') or output.get('file_reference_columns')
    if declared is None:
        columns = output.get('required_output', {}).get('columns')
        if isinstance(columns, list):
            declared = [c for c in columns if c not in keys]
    if declared is None:
        declared = ['restored_image_path']
    if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], str) or not declared[0]:
        return None
    pc = declared[0]
    if tc in keys or pc in keys:
        raise ValueError('Hidden target and prediction columns cannot be sample keys.')
    if tc not in context['answer_columns'] or tc not in context['hidden_df']:
        raise ValueError('Target path must be a declared hidden answer.')
    for name in ('score_df', 'test_df'):
        df = context[name]
        cols = list(dict.fromkeys(keys + [ic] + ([group] if group else [])))
        if any(c not in df for c in cols) or tc in df:
            raise ValueError('Expected rows must expose input/group metadata and hide targets.')
        if df[cols].isna().any().any() or df.duplicated(keys).any() or not len(df):
            raise ValueError('Expected sample metadata must be complete and unique.')
        if group and set(df[group].astype(str)) != set(groups):
            raise ValueError('Each scoring split must contain every declared group.')
    return dict(evaluator_api_version=2, task_type='image_restoration',
                prediction_type='image_file_path', primary_metric=primary,
                secondary_metrics=secondary, maximize=True, key_columns=keys,
                prediction_columns=[pc], answer_columns=[tc],
                file_reference_columns=[pc],
                restoration_evaluation=protocol,
                ssim_parameters=ssim,
                prediction_clipping=[0, dr], zero_mse_policy='error: PSNR is infinite',
                validation_artifact='./working/validation_predictions.csv',
                test_submission='./working/submission.csv',
                score_split='./input/biomaker_splits/score_val.csv')
