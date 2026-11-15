import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def load_paper_index():
    return json.loads((ROOT / 'assets/paper-inputs.json').read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_files(root, files):

    root = Path(root)
    missing = [r['path'] for r in files if not (root / r['path']).is_file()]
    if missing:
        raise FileNotFoundError('Incomplete dataset at ' + str(root) + ': ' + ', '.join(missing))
    receipt = root / '.riblet-files.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    current = {}
    for row in files:
        path = root / row['path']
        stat = path.stat()
        algorithm = 'sha256' if 'sha256' in row else 'md5'
        signature = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         algorithm=algorithm, digest=row[algorithm])
        if 'bytes' in row and stat.st_size != row['bytes']:
            raise ValueError('Dataset size mismatch: ' + str(path))
        if previous.get(row['path']) != signature:
            print('Checking ' + str(path), flush=True)
            h = hashlib.new(algorithm)
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != row[algorithm]:
                raise ValueError('Dataset checksum mismatch: ' + str(path))
        current[row['path']] = signature
    _write_json(receipt, current)
    return current


def check_tensor(path, row):
    import numpy as np

    value = np.load(path, allow_pickle=False)
    dtype = np.dtype('<i4' if row['model'].startswith('bert-') else '<f4')
    if list(value.shape) != row['shape'] or value.dtype != dtype:
        raise ValueError('Paper input shape/dtype mismatch: ' + row['content_id'])
    if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != row['tensor_sha256']:
        raise ValueError('Paper input content differs: ' + row['content_id'])
    if row['model'].startswith('bert-') and int(value[1].sum()) != row['effective_length']:
        raise ValueError('Paper input effective length differs')
    return value


def save_input(directory, row, value):
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (row['content_id'] + '.npy')
    if path.exists():
        check_tensor(path, row)
        return path
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, value, allow_pickle=False)
    try:
        check_tensor(temporary, row)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_bert_inputs(dataset_root, processor, rows, output_dir):
    import numpy as np
    import pyarrow.parquet as pq
    from .preprocessing import load_tokenizer, tokenize_bert

    tokenizer = load_tokenizer(Path(processor) / 'tokenizer.json')
    tables = {}
    for row in rows:
        key = (row['task'], row['split'])
        if key not in tables:
            paths = sorted((Path(dataset_root) / key[0]).glob(key[1] + '-*.parquet'))
            if not paths:
                raise FileNotFoundError('Missing GLUE split: ' + '/'.join(key))
            tables[key] = [r for path in paths for r in pq.read_table(path).to_pylist()]
        data = tables[key]
        matches = [i for i, r in enumerate(data) if str(r['idx']) == row['source_id']]
        if matches != [row['original_row_index_zero_based']]:
            raise ValueError('GLUE source ID/row binding differs: ' + row['content_id'])
        values = tokenize_bert(row['task'], data[matches[0]], tokenizer)
        array = np.stack([values[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')])
        save_input(output_dir, row, array.astype('<i4', copy=False))


def prepare_vit_inputs(dataset_root, processor, rows, output_dir):
    import numpy as np
    from PIL import Image
    from transformers import ViTImageProcessor

    transform = ViTImageProcessor.from_pretrained(str(processor), local_files_only=True)
    with tarfile.open(Path(dataset_root) / 'ILSVRC2012_img_val.tar', 'r:') as archive:
        for row in rows:
            member = archive.getmember(row['source_id'])
            if not member.isfile():
                raise ValueError('ImageNet input is not a regular file')
            with archive.extractfile(member) as stream, Image.open(stream) as image:
                array = transform(images=image.convert('RGB'), return_tensors='np')['pixel_values']
            save_input(output_dir, row, np.asarray(array, dtype='<f4'))


def prepare_paper_inputs(config):
    if config['role'] != 'data':
        raise PermissionError('Inference datasets belong to data1')
    if config['profile'] != 'paper':
        raise ValueError('Paper input preparation requires the paper profile')
    from .assets import verify_preprocessor

    verify_preprocessor(config['model'], config['processor'])
    document = load_paper_index()
    family = config['family']
    rows = [r for r in document['requests'] if r['model'].startswith(family + '-')]
    if family == 'bert':
        from huggingface_hub import snapshot_download

        dataset = document['datasets']['glue']
        root = config['dataset_root'] / 'glue' / dataset['revision']

        if not all((root / r['path']).is_file() for r in dataset['files']):
            snapshot_download(repo_id=dataset['repo_id'], repo_type='dataset',
                              revision=dataset['revision'], local_dir=root)
        verify_files(root, dataset['files'])
        prepare_bert_inputs(root, config['processor'], rows, config['paper_input_dir'])
    else:
        dataset = document['datasets']['imagenet']
        root = config['dataset_root'] / 'imagenet-1k'
        try:
            verify_files(root, dataset['files'])
        except FileNotFoundError as error:
            raise FileNotFoundError(str(error) + '\nObtain the complete archives with your own '
                                    'ImageNet authorization: ' + dataset['homepage']) from error
        prepare_vit_inputs(root, config['processor'], rows, config['paper_input_dir'])
    index_path = config['paper_input_dir'] / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index.update({r['content_id']: r for r in rows})
    _write_json(index_path, index)
    return dict(status='paper-inputs-ready', family=family, prepared=len(rows),
                dataset_complete=True, directory=str(config['paper_input_dir']))


def resolve_paper_input(config):
    if config['role'] != 'data':
        raise PermissionError('Inference input belongs to data1')

    if config.get('inputs') is not None:
        return config['inputs'], dict(kind='user-input')
    sample = config.get('paper_sample')
    rows = [r for r in load_paper_index()['requests'] if r['content_id'] == sample]
    if len(rows) != 1 or rows[0]['model'] != config['model'] or config['profile'] != 'paper':
        raise ValueError('Paper sample/model/profile mismatch')
    row = rows[0]
    directory = config['directory'] / 'inputs/paper'
    index_path = directory / 'index.json'
    if not index_path.exists() or json.loads(index_path.read_text()).get(sample) != row:
        raise ValueError('Run the data-owner inputs command before inference')
    path = directory / (sample + '.npy')
    check_tensor(path, row)
    return path, dict(row, kind='paper-sample-reproduction')
