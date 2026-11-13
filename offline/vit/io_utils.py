import hashlib,json,os
from pathlib import Path
import numpy as np

from abi import GRAPH_ID


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)


def manifest(root, custom=False):
    from riblet.development import image_identity
    doc=json.loads((Path(root)/'manifest.json').read_bytes())
    return doc['rows'],image_identity(doc,custom=custom)


def model_and_processor(root,preprocessor=None):
    import jax
    from flax.serialization import msgpack_restore
    from transformers import ViTImageProcessor
    import model as m
    root=Path(root);raw=msgpack_restore((root/'flax_model.msgpack').read_bytes())
    params=jax.tree_util.tree_map(m.encode,raw)
    return params,ViTImageProcessor.from_pretrained(str(preprocessor or root),local_files_only=True)


def pixels(data_root,row,processor):
    from PIL import Image
    import model as m
    with Image.open(Path(data_root)/'images'/row['relative_id']) as im:
        value=processor(images=im.convert('RGB'),return_tensors='np')['pixel_values']
    return m.encode(value.transpose(0,2,3,1))


def pixels_host(data_root,row,processor):

    import io
    from PIL import Image
    raw=(Path(data_root)/'images'/row['relative_id']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Development image content differs from its identity')
    with Image.open(io.BytesIO(raw)) as im:
        value=processor(images=im.convert('RGB'),return_tensors='np')['pixel_values']
    if value.shape!=(1,3,224,224) or not np.isfinite(value).all():raise ValueError("unsupported ViT processor output")
    return np.trunc(value.transpose(0,2,3,1).astype(np.float64)*(1<<18)).astype(np.int64)


def residual(op,x,y):
    x=np.asarray(x,np.float64)/(1<<18);y=np.asarray(y,np.float64)/(1<<18)
    return np.abs(1-(x if op=='reciprocal' else np.sqrt(x))*y)


def digest(document):return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
