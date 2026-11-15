import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from .population import freeze_identities, SELECTION_POLICY


def select_members(tar,synset):
    members=[];seen=set()
    for m in tar:
        if not m.isfile():continue
        p=PurePosixPath(m.name)
        if len(p.parts)!=1 or not re.fullmatch(re.escape(synset)+r'_[0-9]+\.JPEG',m.name):
            raise ValueError('unexpected non-train image identity in class tar')
        if m.name in seen:raise ValueError('duplicate image identity in source class')
        seen.add(m.name);members.append(m)
    if len(members)<10:raise ValueError('source class has fewer than10 images')
    return sorted(members,key=lambda m:m.name)[:10],len(members)


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def extract_class(archive,root,binding,class_id,synset,member):
    checkpoint=root/'classes'/f'{synset}.json'
    if checkpoint.exists():
        prior=json.loads(checkpoint.read_text())
        if prior['source_binding']!=binding:raise ValueError('source archive identity changed')
        for r in prior['records']:
            path=root/'images'/r['relative_id']
            if not path.is_file() or path.stat().st_size!=r['bytes']:
                raise ValueError('incomplete prior image checkpoint')
        return prior


    with Path(archive).open('rb',buffering=0) as raw, tarfile.open(fileobj=raw,mode='r:') as outer, outer.extractfile(member) as file, tarfile.open(fileobj=file,mode='r:') as inner:
        selected,count=select_members(inner,synset)
        destination=root/'images'/synset;destination.mkdir(exist_ok=True)
        rows=[]
        for image in selected:
            with inner.extractfile(image) as f:data=f.read()
            with Image.open(io.BytesIO(data)) as img:
                width,height=img.size;mode=img.mode;img.verify()
            path=destination/image.name
            tmp=path.with_suffix('.JPEG.tmp');tmp.write_bytes(data);tmp.replace(path)
            rows.append(dict(split='train',relative_id=f'{synset}/{image.name}',
                synset=synset,class_id=class_id,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                width=width,height=height,original_mode=mode))
    result=dict(source_binding=binding,source_class_images=count,records=rows)
    atomic_json(checkpoint,result)
    return result


def prepare(archive,output):
    archive=Path(archive);root=Path(output)
    if archive.name!='ILSVRC2012_img_train.tar':raise ValueError('only official training archive name accepted')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('population already frozen; do not regenerate')
    (root/'classes').mkdir(exist_ok=True);(root/'images').mkdir(exist_ok=True)
    stat=archive.stat();binding=dict(source=str(archive),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                   selection_policy=SELECTION_POLICY)
    records=[];source_count=0;start=time.monotonic()
    with tarfile.open(archive,'r:') as outer:
        classes={m.name[:-4]:m for m in outer if m.isfile() and re.fullmatch(r'n[0-9]{8}\.tar',m.name)}
        if len(classes)!=1000:raise ValueError('not the1000-class ImageNet training archive')
    workers=int(os.environ.get('VIT_DATA_WORKERS','8'))
    if not 1<=workers<=32:raise ValueError('data-reader workers must be in1..32')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(extract_class,archive,root,binding,i,s,classes[s]) for i,s in enumerate(sorted(classes))]
        for completed,future in enumerate(as_completed(jobs),1):
            current=future.result();records.extend(current['records']);source_count+=current['source_class_images']
            if completed%25==0 or completed==1000:
                atomic_json(root/'progress.json',dict(completed_classes=completed,total_classes=1000,
                            selected_images=len(records),elapsed_seconds=time.monotonic()-start,status='preparing'))
                print(json.dumps(dict(classes=completed,images=len(records))),flush=True)
    frozen=freeze_identities(records);by_id={r['relative_id']:r for r in records}
    complete=[dict(by_id[r['relative_id']],**{k:r[k] for k in ('request_ordinal','fold_id','class_rank')}) for r in frozen]
    hashes=Counter(r['sha256'] for r in complete)
    result=dict(schema='vit-development-train10k-v1',source_binding=binding,source_train_image_count=source_count,
                selection_policy=SELECTION_POLICY,rows=complete,request_count=10000,
                fold_counts=[sum(r['fold_id']==f for r in complete) for f in range(5)],
                content_duplicate_excess=sum(v-1 for v in hashes.values()),
                official_validation_read=False,model_outputs_read=False)
    if source_count!=1281167:raise ValueError('unexpected official training population size')
    atomic_json(root/'manifest.json',result)
    digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    atomic_json(root/'READY.json',dict(status='frozen_train10k_ready',manifest_sha256=digest,
                bytes=sum(r['bytes'] for r in complete),requests=10000,fold_counts=result['fold_counts'],
                model_forwards=0,official_validation_read=False))
    print(json.dumps(dict(status='complete',manifest_sha256=digest,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.archive,a.output)
