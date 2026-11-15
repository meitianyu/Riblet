from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,pickle,time
import numpy as np
from baseline_io import sha,write_json
from cdi_bert.batch import precision_pass_batch
from experiment_contract import (LINEAGES,SEED_LINEAGES,ALGORITHM_VERSION,DESIGN_BITS,
    DESIGN_INCLUSIVE,NUMERIC_REFERENCE,GRAPH_ID,design_pass)
from public_sites import SCHEMA
from prepared_store import PreparedStore
from runtime import request_queues,restore_order
from layout import PHYSICAL_METADATA,physical_length

GRAPH_CODE_HASH=hashlib.sha256(b''.join(Path(__file__).with_name(n).read_bytes()
    for n in ('model.py','causal_graph.py','runtime.py','plaintext_numeric.py','native_mixed.py','layout.py','experiment_contract.py'))+GRAPH_ID.encode()).hexdigest()

def load_profile(path):
    doc=json.loads(Path(path).read_bytes())
    if doc['schema']!=SCHEMA or doc['layout']!='pad128':raise ValueError('incompatible site table')
    return doc

def merge_profiles(documents,path):
    if not documents or len({d['design'] for d in documents})!=1:raise ValueError('mixed population')
    merged=dict(schema=SCHEMA,design=documents[0]['design'],layout='pad128',sites=[r for d in documents for r in d['sites']])
    from public_sites import write_sites
    write_sites(path,merged);return merged

def chunks(items,size=32):
    for start in range(0,len(items),size):yield items[start:start+size]

def write_operands(path,rows,indices,operands,*,heads):
    selected=[rows[i] for i in indices]
    if len(selected)!=len(operands):raise ValueError('arrived operand/request count mismatch')
    values=np.concatenate([np.asarray(v,np.int64).reshape(heads,physical_length(r['effective_length']))[:,:r['effective_length']] for r,v in zip(selected,operands)],axis=1) if selected else np.empty((heads,0),np.int64)
    np.savez_compressed(path,operands=values,request_ids=np.asarray([r['request_ordinal'] for r in selected],np.int64),
        folds=np.asarray([r['fold_id'] for r in selected],np.uint8),lengths=np.asarray([r['effective_length'] for r in selected],np.int32))

def update_ledger(ledger,point,record):
    operator='reciprocal' if point>0 and (point-1)%3==0 else 'rsqrt'
    heads=12 if operator=='reciprocal' else 1;n=ledger['effective_length']
    physical=physical_length(n)
    px=np.asarray(record['operand'],np.int64).reshape(heads,physical);py=np.asarray(record['inverse'],np.int64).reshape(heads,physical)
    x=px[:,:n].reshape(-1);y=py[:,:n].reshape(-1)
    normal=np.asarray(record['normal'],bool).reshape(heads,physical)[:,:n].reshape(-1)
    k=np.asarray(record['k'],np.int64).reshape(heads,physical)[:,:n].reshape(-1)
    ledger[operator+'_physical_arrived']+=px.size
    physical_pass=precision_pass_batch(px.reshape(-1),py.reshape(-1),operator=operator,bits=12,inclusive=True)
    ledger[operator+'_physical_p12_failed']+=int((~physical_pass).sum())
    ledger[operator+'_arrived']+=len(x);ledger[operator+'_normal']+=int(normal.sum())
    ledger[operator+'_normal_k_sum']+=int(k[normal].sum())
    p12=precision_pass_batch(x,y,operator=operator,bits=12,inclusive=True)
    p13=precision_pass_batch(x,y,operator=operator,bits=13,inclusive=False)
    design=design_pass(operator,x,y)
    ledger['normal_design_mismatches']+=int(np.count_nonzero(normal&~design))
    ledger['normal_p13_diagnostic_exceedances']+=int(np.count_nonzero(normal&~p13))
    ledger['p12_pass']=ledger['p12_pass'] and bool(np.all(p12))
    if int(record['flags']) or np.any(px<=0) or np.any(py<=0):
        bad=np.flatnonzero(((px<=0)|(py<=0)).reshape(-1))
        detail={'point':point,'phase':'inverse_or_finish','flags':int(record['flags'])}
        if len(bad):
            idx=int(bad[0]);detail.update(head=idx//physical,token_position=idx%physical,is_valid_token=idx%physical<n,
                operand_encoded=int(px.reshape(-1)[idx]),inverse_encoded=int(py.reshape(-1)[idx]))
        ledger.update(status='local_invalid',first_failure=detail,p12_pass=False)
    else:
        xd=x.astype(np.float64)/(1<<18);yd=y.astype(np.float64)/(1<<18)
        error=float(np.max(np.abs(1-(xd if operator=='reciprocal' else np.sqrt(xd))*yd)))
        ledger[operator+'_max_residual']=max(ledger[operator+'_max_residual'],error)
        pxd=px.astype(np.float64)/(1<<18);pyd=py.astype(np.float64)/(1<<18)
        pe=float(np.max(np.abs(1-(pxd if operator=='reciprocal' else np.sqrt(pxd))*pyd)))
        ledger[operator+'_physical_max_residual']=max(ledger[operator+'_physical_max_residual'],pe)

def checkpoint(root,point,states,ledgers,profiles,model_hash,previous):
    folder=root/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/f'after-{point:02d}.pkl';temporary=path.with_suffix('.tmp')
    for leftover in (path,temporary):
        if leftover.exists():leftover.rename(folder/f'orphan-{leftover.name}-{time.time_ns()}')
    digest=hashlib.sha256()
    class HashWriter:
        def __init__(self,stream):self.stream=stream
        def write(self,data):digest.update(data);return self.stream.write(data)
    with temporary.open('xb') as f:
        pickle.dump(dict(point=point,states=states,ledgers=ledgers,profiles=profiles,model_hash=model_hash,
            algorithm_version=ALGORITHM_VERSION,design_bits=DESIGN_BITS,numeric_reference=NUMERIC_REFERENCE,
            graph_id=GRAPH_ID,graph_code_hash=GRAPH_CODE_HASH),HashWriter(f),protocol=5)
        f.flush();os.fsync(f.fileno())
    temporary.rename(path)
    write_json(folder/'latest.json',dict(path=path.name,sha256=digest.hexdigest(),point=point,model_hash=model_hash,profiles=profiles,graph_id=GRAPH_ID))
    if previous is not None:
        old=Path(previous).resolve()
        if old.parent!=folder.resolve() or not old.name.startswith('after-') or old.suffix!='.pkl':raise ValueError('unsafe checkpoint retirement')
        old.unlink() 
    return path

def semantic_record(record,n,heads):

    out=dict(record)
    for key in ('operand','inverse','normal','k','mean','mean_square','raw_variance',
                'z_index','normalized','polynomial','compensation_word'):
        if key in out:out[key]=np.asarray(out[key]).reshape(heads,physical_length(n))[:,:n].copy()
    return out

def retire_finished_checkpoint(root):
    root=Path(root)
    if not (root/'result.json').exists() or json.loads((root/'result.json').read_text()).get('status')!='causal_task_complete':
        raise ValueError('task is not complete')
    marker=root/'checkpoints/latest.json'
    if not marker.exists():return
    meta=json.loads(marker.read_text())
    if meta.get('status')=='retired_after_task_completion':return
    path=(marker.parent/meta['path']).resolve()
    if path.parent!=marker.parent.resolve() or not path.name.startswith('after-') or path.suffix!='.pkl':raise ValueError('invalid checkpoint path')
    if sha(path)!=meta['sha256']:raise ValueError('checkpoint integrity mismatch')
    path.unlink()
    write_json(marker,dict(status='retired_after_task_completion',retired_path=meta['path'],
        recoverable_from='frozen model, point operands, profiles and request metadata',task_result_sha256=sha(root/'result.json')))

def run_task(args,root,mode,task,rows,params,model_hash,kernels,cpu,status):
    import causal_graph as graph
    import runtime
    task_started=time.perf_counter();task_root=root/mode/task;task_root.mkdir(parents=True,exist_ok=True)
    if (task_root/'result.json').exists():
        result=json.loads((task_root/'result.json').read_text());document=load_profile(task_root/'tables.json')
        if result['table_sha256']!=sha(task_root/'tables.json'):raise ValueError('completed task table changed')
        retire_finished_checkpoint(task_root)
        return result,document
    states=[None]*len(rows);profiles={};documents=[];old_checkpoint=None
    ledgers=[]
    for r in rows:
        item={k:r[k] for k in ('request_ordinal','request_digest','fold_id','effective_length')}
        item.update(status='active',first_failure=None,p12_pass=True,logits_encoded=None,
                    normal_design_mismatches=0,normal_p13_diagnostic_exceedances=0)
        for op in ('reciprocal','rsqrt'):item.update({op+'_arrived':0,op+'_normal':0,op+'_normal_k_sum':0,op+'_max_residual':0.,
            op+'_physical_arrived':0,op+'_physical_p12_failed':0,op+'_physical_max_residual':0.})
        ledgers.append(item)
    targets=graph.target_points(mode)
    start_point=0
    marker=task_root/'checkpoints/latest.json'
    if marker.exists():
        meta=json.loads(marker.read_text());path=marker.parent/meta['path']
        if sha(path)!=meta['sha256']:raise ValueError('resume checkpoint hash mismatch')
        with path.open('rb') as f:saved=pickle.load(f)
        if saved.get('algorithm_version')!=ALGORITHM_VERSION or saved.get('design_bits')!=DESIGN_BITS or saved.get('numeric_reference')!=NUMERIC_REFERENCE or saved.get('graph_id')!=GRAPH_ID or saved.get('graph_code_hash')!=GRAPH_CODE_HASH:
            raise ValueError('checkpoint belongs to another family or gate')
        if saved['model_hash']!=model_hash:raise ValueError('checkpoint model changed')
        states=saved['states'];ledgers=saved['ledgers'];profiles=saved['profiles'];start_point=saved['point']+1;old_checkpoint=path
        for point in targets:
            if point<start_point:
                table=task_root/f'point-{point:02d}/fit/tables.json'
                if sha(table)!=profiles[str(point)]:raise ValueError('checkpoint profile changed')
                documents.append(load_profile(table))
        retained=task_root/f'retained-after-checkpoint-{time.time_ns()}'
        later=[d for d in task_root.glob('point-*') if int(d.name.split('-')[1])>=start_point]
        if later:
            retained.mkdir()
            for d in later:d.rename(retained/d.name)
        write_json(task_root/'resume.json',dict(start_point=start_point,checkpoint=str(path),retained=str(retained)))
    checkpoint_pool=ThreadPoolExecutor(max_workers=1)
    checkpoint_future=None
    for point in range(start_point,args.stop_after_point+1):
        begin=time.perf_counter();point_root=task_root/f'point-{point:02d}';point_root.mkdir()
        active=[i for i,v in enumerate(ledgers) if v['status']=='active'];target=point in targets
        status(mode,task,point,'prepare' if target else 'fixed_spu',active_requests=len(active))
        records=[];record_indices=[];arrived=[];operands=[]
        if target:
            store=PreparedStore(point_root/'prepared-shards',dict(graph=GRAPH_ID,graph_code_hash=GRAPH_CODE_HASH,
              task=task,point=point,model=model_hash,population=args.population_id,upstream=dict(profiles)),kind=runtime.kind(point))
            for queue_number,queue in enumerate(request_queues(active,rows,size=32)):
                results=kernels.prepare_many(point,[states[i] for i in queue],params,[rows[i] for i in queue])
                kept=[];kept_ids=[]
                for i,prep in zip(queue,results):
                    states[i]=None
                    if int(prep['flags']):
                        ledgers[i].update(status='local_invalid',first_failure={'point':point,'phase':'before_inverse','flags':int(prep['flags'])},p12_pass=False)
                    else:
                        kept.append(prep);kept_ids.append(i);arrived.append(i);operands.append(prep['operand'].value.copy())
                store.put(kept_ids,kept)
                del results,kept
                if queue_number%16==0:
                    status(mode,task,point,'prepare',active_requests=len(active),prepared_requests=len(arrived),
                           prepared_disk_bytes=store.stats['written_bytes'],memory_bounded_queue=32)
            arrived,operands=restore_order(arrived,operands)
            write_operands(point_root/'operands.npz',rows,arrived,operands,heads=12 if (point>0 and (point-1)%3==0) else 1)
            del operands
            write_json(point_root/'input.json',dict(lineage=LINEAGES[mode],task=task,point=point,model_hash=model_hash,
                graph_id=GRAPH_ID,graph_code_hash=GRAPH_CODE_HASH,physical_length=PHYSICAL_METADATA,
                algorithm_version=ALGORITHM_VERSION,design_bits=DESIGN_BITS,design_inclusive=DESIGN_INCLUSIVE,
                numeric_reference=NUMERIC_REFERENCE,numeric_seed_lineage_id=SEED_LINEAGES[LINEAGES[mode]],
                population_sha256=args.population_id,
                upstream_profiles=dict(profiles),total_requests=len(rows),arrived_requests=len(arrived),
                inherited_invalid_requests=len(rows)-len(active),local_invalid_before_inverse=len(active)-len(arrived),
                sha256=sha(point_root/'operands.npz')))
            status(mode,task,point,'cpu_fit',active_requests=len(arrived))
            document,fit=cpu.fit(point_root,mode,task,point)
            documents.append(document);profiles[str(point)]=fit['table_sha256']
            enabled=any(r['enabled'] for r in document['sites'])
            status(mode,task,point,'apply_frozen',active_requests=len(arrived))
            applied_ids=[]
            for queue,prepared,shard_path in store.iter_shards():
                choices=[graph.selection_arrays(document,rows[i],point) if enabled else None for i in queue]
                results=kernels.finish_many(point,prepared,params,choices)
                for i,(hidden,record) in zip(queue,results):
                    update_ledger(ledgers[i],point,record)
                    if ledgers[i]['normal_design_mismatches']:
                        raise ValueError(f'normal p12 mismatch: request={rows[i]["request_ordinal"]},point={point}')
                    if ledgers[i]['status']=='active':states[i]=hidden
                    records.append(semantic_record(record,rows[i]['effective_length'],12 if runtime.kind(point)==1 else 1));record_indices.append(i)
                applied_ids.extend(queue);store.consume(shard_path);del prepared,results,choices
                if len(applied_ids)%512<32:
                    status(mode,task,point,'apply_frozen',active_requests=len(arrived),applied_requests=len(applied_ids),memory_bounded_queue=32)
            if sorted(applied_ids)!=arrived:raise ValueError('bounded prepared application lost or duplicated a request')
            write_json(point_root/'prepared-storage-audit.json',dict(**store.stats,all_shards_consumed=True,
              numeric_values_unchanged=True,only_dead_states_omitted=['exps','value'] if runtime.kind(point)==1 else ['centered']))
        else:
            raise ValueError("expected a public-site inverse point")
        record_indices,records=restore_order(record_indices,records)
        arrays={'request_ids':np.asarray([rows[i]['request_ordinal'] for i in record_indices],np.int64),
                'lengths':np.asarray([rows[i]['effective_length'] for i in record_indices],np.int32)}
        for key in ('operand','inverse','normal','k'):
            dtype=np.bool_ if key=='normal' else np.int64
            arrays[key]=np.concatenate([np.asarray(r[key],dtype=dtype).reshape(-1) for r in records]) if records else np.empty(0,dtype)
        if point==0 or (point-1)%3!=0:
            for field in ('mean','mean_square','raw_variance'):
                arrays[field]=np.concatenate([np.asarray(r[field],np.int64).reshape(-1) for r in records]) if records else np.empty(0,np.int64)
            arrays['spu_trace_present']=~arrays['normal']
            for key in ('z_index','normalized','polynomial','compensation_word'):
                raw=np.concatenate([np.asarray(r[key],np.int64).reshape(-1) for r in records]) if records else np.empty(0,np.int64)
                arrays[key]=np.where(arrays['spu_trace_present'],raw,0)
        np.savez_compressed(point_root/'outputs.npz',**arrays)
        n_original=sum(r['effective_length'] for r in rows)*(12 if point>0 and (point-1)%3==0 else 1)
        write_json(point_root/'result.json',dict(status='point_frozen_and_propagated' if target else 'fixed_spu_complete',point=point,
            theoretical_events=n_original,arrived_events=len(arrays['operand']),unreached_events=n_original-len(arrays['operand']),
            graph_id=GRAPH_ID,physical_length=PHYSICAL_METADATA,
            physical_theoretical_events=sum(physical_length(r['effective_length']) for r in rows)*(12 if runtime.kind(point)==1 else 1),
            physical_arrived_events=sum(physical_length(rows[i]['effective_length']) for i in record_indices)*(12 if runtime.kind(point)==1 else 1),
            semantic_scope='effective_tokens_only;PAD_is_physical_native_fallback_not_fit_data',
            inherited_invalid_requests=len(rows)-len(active),local_invalid_before_inverse=len(active)-len(arrived) if target else 0,
            fit_table_sha256=profiles.get(str(point)),
            normal_events=int(arrays['normal'].sum()),surviving_requests=sum(v['status']=='active' for v in ledgers),
                seconds=time.perf_counter()-begin,outputs_sha256=sha(point_root/'outputs.npz'),
                rsqrt_epsilon_encoded=0,spu_compensation_scale_bits=27,spu_final_shift=27,
                trace_mask_semantics='spu_trace_present=false means compensation fields are null, not zero-valued data'))
        write_json(task_root/'requests.json',ledgers)
        del arrays,records
        if target:
            if checkpoint_future is not None:old_checkpoint=checkpoint_future.result()
            checkpoint_future=checkpoint_pool.submit(checkpoint,task_root,point,list(states),
                [dict(v) for v in ledgers],dict(profiles),model_hash,old_checkpoint)
        status(mode,task,point,'point_complete',active_requests=sum(v['status']=='active' for v in ledgers))
    if checkpoint_future is not None:old_checkpoint=checkpoint_future.result()
    checkpoint_pool.shutdown(wait=True)
    if args.stop_after_point < 36:
        document=merge_profiles(documents,task_root/'fixture-tables.json')
        result=dict(status='fixture_complete',point=args.stop_after_point,requests=len(rows),
                    deployment_allowed=False,table_sha256=sha(task_root/'fixture-tables.json'))
        write_json(task_root/'fixture-result.json',result)
        return result,document
    for queue in chunks([i for i,v in enumerate(ledgers) if v['status']=='active']):
        head={'bert':{'pooler':params['bert']['pooler']},'classifier':params['classifier']}
        results=kernels.classify_many([states[i][:,0:1,:] for i in queue],head,[rows[i] for i in queue])
        for i,(logits,flags) in zip(queue,results):
            if int(flags):ledgers[i].update(status='local_invalid',first_failure={'point':37,'phase':'classifier','flags':int(flags)},p12_pass=False)
            else:ledgers[i].update(status='complete',logits_encoded=np.asarray(logits).reshape(-1).tolist())
    for item in ledgers:
        item['error_status']='finite' if item['status']=='complete' else 'positive_infinity'
        for op,factor in (('reciprocal',144),('rsqrt',25)):
            item[op+'_unreached']=factor*item['effective_length']-item[op+'_arrived']
            item[op+'_physical_unreached']=factor*physical_length(item['effective_length'])-item[op+'_physical_arrived']
            if item['status']!='complete':item[op+'_max_residual']=None
    write_json(task_root/'requests.json',ledgers)
    merged=merge_profiles(documents,task_root/'tables.json')
    expected=(144 if mode=='reciprocal' else 25 if mode=='rsqrt' else 169)
    if len(merged['sites'])!=expected:raise ValueError('terminal task table registry incomplete')
    result=dict(status='causal_task_complete',mode=mode,lineage=LINEAGES[mode],task=task,requests=len(rows),
        numeric_invalid_count=sum(v['status']!='complete' for v in ledgers),p12_passing_requests=sum(v['p12_pass'] for v in ledgers),
        normal_design_mismatches=sum(v['normal_design_mismatches'] for v in ledgers),
        normal_p13_diagnostic_exceedances=sum(v['normal_p13_diagnostic_exceedances'] for v in ledgers),
        design_bits=DESIGN_BITS,design_inclusive=DESIGN_INCLUSIVE,algorithm_version=ALGORITHM_VERSION,sites=expected,points=len(profiles),
        graph_id=GRAPH_ID,graph_code_hash=GRAPH_CODE_HASH,physical_length=PHYSICAL_METADATA,
        elapsed_seconds=time.perf_counter()-task_started,table_sha256=sha(task_root/'tables.json'))
    write_json(task_root/'result.json',result)
    retire_finished_checkpoint(task_root)
    return result,merged
