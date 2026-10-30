from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import urllib.request
import zipfile

from .assets import model_fingerprint
from .core import Store,digest,identifier,now,settings,write_json


class Cancelled(Exception):pass


def resources_ready(gpu):
    try:
        health=json.load(urllib.request.urlopen(os.environ.get('WORKBENCH_INFERENCE_HEALTH','http://127.0.0.1:8711/health'),timeout=3)) if os.environ.get('WORKBENCH_SHARED_GPU','1') == '1' else {'activeBatches':{},'queuedBatches':0}
        if health['activeBatches'] or health['queuedBatches']:return False
        uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        return not any(line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid() for line in processes.splitlines())
    except Exception:return False


from workbench_server.gpu_lock import leased

@leased
def run_job(store,job,config):
    directory=store.root/'jobs'/job['id'];manifest=json.loads((directory/'manifest.json').read_text())
    parent=store.get('models',manifest['parent_model_id']);events=directory/'events.jsonl'
    def callback(stage,progress=None):
        if store.job(job['id']).get('cancel_requested'):raise Cancelled()
        store.update_job(job['id'],state=stage,progress=progress or {})
        with events.open('a') as file:file.write(json.dumps({'time':now(),'stage':stage,'progress':progress or {}})+'\n')
    engine=None
    try:
        if shutil.disk_usage(store.root).free<24*1024**3:raise ValueError('完整权重导出与打包至少需要24GiB可用空间')
        source_dir=Path(__file__).parent
        snapshot=directory/'source';snapshot.mkdir()
        for p in source_dir.rglob('*.py'):
            destination=snapshot/p.relative_to(source_dir);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
        write_json(directory/'source_manifest.json',{str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*.py')})
        fingerprint,_=model_fingerprint(parent['path'])
        if fingerprint!=parent['fingerprint']:raise ValueError('父权重指纹变化，拒绝在可变输入上训练')
        for collection in manifest['sets'].values():
            for row in collection:
                if digest(row['image'])!=row['image_sha256'] or digest(row['label'])!=row['label_sha256']:
                    raise ValueError('输入数据指纹变化')
        from .engine import Sam3Engine
        request=manifest['request'];callback('BASELINING')
        engine=Sam3Engine(parent['path'],request['resolution'],request['seed'])
        baseline=engine.evaluate(manifest['sets']['eval'],directory/'baseline_predictions.jsonl',callback,'BASELINING')
        write_json(directory/'baseline.json',baseline)
        callback('TRAINING');training=engine.train(manifest,directory,callback)
        callback('EXPORTING');engine.export(directory/'model',config['base'])
        engine.close();engine=None
        callback('VERIFYING')
        # A new model instance reconstructs exclusively from the exported files.
        engine=Sam3Engine(directory/'model',request['resolution'],request['seed'])
        if any('lora_' in name for name in engine.model.state_dict()):raise ValueError('导出模型仍依赖LoRA包装')
        callback('EVALUATING')
        candidate=engine.evaluate(manifest['sets']['eval'],directory/'candidate_predictions.jsonl',callback,'EVALUATING')
        engine.close();engine=None
        deltas={name:candidate[name]['f1']-baseline[name]['f1'] for name in candidate}
        regressions=[name for name,delta in deltas.items() if delta<0]
        invalid=sum(value['invalid_boxes'] for value in candidate.values())
        report={'baseline':baseline,'exported_model':candidate,'f1_deltas':deltas,'regressions':regressions,
            'new_classes':manifest['new_classes'],'old_classes':manifest['old_classes'],
            'training_mode':manifest.get('training_mode','replay'),
            'old_knowledge_status':manifest.get('old_knowledge_status','evaluated_subset'),
            'evaluated_old_classes':manifest.get('evaluated_old_classes',manifest['old_classes']),
            'quality_status':'needs_review','automatic_quality_approval':False,
            'reason':'尚未设定业务验收阈值；候选权重可加载不代表全部类别质量通过',
            'candidate_invalid_boxes':invalid,'warnings':manifest['warnings'],
            'evaluated_exported_complete_weights':True,'training_scope':'bbox_text_detection','segmentation_quality':'not_evaluated','training':training}
        write_json(directory/'evaluation.json',report)
        write_json(directory/'class_prompts.json',manifest['classes']);write_json(directory/'training_config.json',request)
        if model_fingerprint(parent['path'])[0]!=parent['fingerprint']:raise ValueError('父权重发生变化')
        for row in manifest['sets']['eval']:
            if digest(row['image'])!=row['image_sha256']:raise ValueError('评测期间输入指纹变化')
        version_id=identifier('model');weight_fp,hashes=model_fingerprint(directory/'model')
        metadata={'schema_version':1,'model_version_id':version_id,'parent_model_version_id':parent['id'],
             'parent_model_fingerprint':parent['fingerprint'],
             'training_mode':report['training_mode'],'old_knowledge_status':report['old_knowledge_status'],
             'model_family':'sam3','artifact_type':'complete_merged_model','standalone':True,
             'weight_fingerprint':weight_fp,'weight_hashes':hashes,'classes':manifest['classes'],
             'replay_asset_ids':manifest['next_replay_assets'],'job_id':job['id'],'quality_status':'needs_review'}
        write_json(directory/'result_manifest.json',metadata)
        callback('PACKAGING')
        package=directory/'weights.zip';members={}
        with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_STORED) as archive:
            for folder in ['model','adapter']:
                for p in sorted((directory/folder).rglob('*')):
                    if p.is_file():
                        callback('PACKAGING',{'file':str(p.relative_to(directory))})
                        name=p.relative_to(directory).as_posix();archive.write(p,name);members[name]=digest(p)
            for name in ['evaluation.json','class_prompts.json','training_config.json','result_manifest.json','training.json','source_manifest.json']:
                p=directory/name;archive.write(p,name);members[name]=digest(p)
            archive.writestr('file_checksums.json',json.dumps(members,indent=2))
        callback('PACKAGING',{'checking':'archive CRC and export hashes'})
        with zipfile.ZipFile(package) as archive:
            bad=archive.testzip()
            if bad:raise ValueError('导出ZIP CRC失败：'+bad)
        artifacts={}
        for name in ['weights.zip','evaluation.json','class_prompts.json','training_config.json','result_manifest.json','events.jsonl']:
            p=directory/name;artifacts[name]={'path':str(p),'sha256':digest(p),'size':p.stat().st_size}
        version={'id':version_id,'name':request.get('name') or '增量候选 '+job['id'][-8:],'path':str(directory/'model'),
            'parent_model_fingerprint':parent['fingerprint'],
            'training_mode':report['training_mode'],'old_knowledge_status':report['old_knowledge_status'],
            'fingerprint':weight_fp,'weight_hashes':hashes,'family':'sam3','classes':manifest['classes'],
            'replay_assets':manifest['next_replay_assets'],'parent_id':parent['id'],'job_id':job['id'],
            'created_utc':now(),'status':'candidate','quality_status':'needs_review'}
        # Commit model and final job in one transaction; cancellation cannot race publication.
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=json.loads(db.execute('SELECT data FROM jobs WHERE id=?',(job['id'],)).fetchone()['data'])
            if current.get('cancel_requested'):raise Cancelled()
            db.execute('INSERT INTO objects VALUES (?,?,?)',('models',version_id,json.dumps(version,ensure_ascii=False)))
            current.update(state='COMPLETED',progress={'complete':True},finished_utc=now(),updated_utc=now(),
                       result_model_version_id=version_id,artifacts=artifacts,evaluation=report)
            db.execute('UPDATE jobs SET state=?,data=? WHERE id=?',('COMPLETED',json.dumps(current,ensure_ascii=False),job['id']))
    except Cancelled:
        store.update_job(job['id'],state='CANCELLED',finished_utc=now(),error='任务已取消，未发布新权重版本')
    except Exception as exc:
        (directory/'failure.log').write_text(traceback.format_exc())
        store.update_job(job['id'],state='FAILED',finished_utc=now(),error=str(exc))
    finally:
        if engine is not None:engine.close()


def main():
    import fcntl
    config=settings();store=Store(config['root'])
    lock=(store.root/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store.recover()
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        for job in store.jobs():
            if job['state'] in {'BASELINING','TRAINING','EXPORTING','VERIFYING','EVALUATING','PACKAGING'}:store.cancel(job['id'])
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopping:
        if resources_ready(config['gpu']):
            job=store.claim()
            if job:run_job(store,job,config);continue
        time.sleep(2)


if __name__=='__main__':main()
