import hashlib,tempfile,unittest
from pathlib import Path

class FetchTests(unittest.TestCase):
    def test_http_errors_html_and_bad_hash_not_cached(self):
        import threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from riblet.resource_fetch import fetch_resource
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_GET(self):
                code=int(self.path[1:]) if self.path[1:].isdigit() else 200
                self.send_response(code)
                if self.path=='/html':self.send_header('Content-Type','text/html')
                self.end_headers();self.wfile.write(b'bad')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for path in ('403','429','html','bad'):
                with tempfile.TemporaryDirectory() as d:
                    h=hashlib.sha256(b'abc').hexdigest()
                    r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[f'http://127.0.0.1:{server.server_port}/{path}'])
                    with self.assertRaises(ValueError):fetch_resource(r,d)
                    self.assertFalse((Path(d)/h).exists())
                    from filelock import FileLock

                    with FileLock(str(Path(d)/(h+'.lock')), timeout=0):pass
        finally:server.shutdown();server.server_close();thread.join()

    def test_local_import_obeys_download_lock(self):
        from riblet.resource_fetch import fetch_resource
        from filelock import FileLock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_bytes(b'abc')
            h=hashlib.sha256(b'abc').hexdigest()
            r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[])
            with FileLock(str(root/(h+'.lock')),timeout=0):
                with self.assertRaises(FileExistsError):fetch_resource(r,root,local_file=source)

    def test_orphan_download_lock_does_not_block_local_import(self):
        from riblet.resource_fetch import fetch_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_bytes(b'abc')
            h=hashlib.sha256(b'abc').hexdigest();(root/(h+'.lock')).touch()
            r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[])
            try:receipt=fetch_resource(r,root,local_file=source)
            except FileExistsError:self.fail('a dead downloader still blocks acquisition')
            self.assertEqual(Path(receipt['path']).read_bytes(),b'abc')

    def test_terminated_lock_owner_releases_resource_for_retry(self):
        import subprocess,sys
        from riblet.resource_fetch import fetch_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_bytes(b'abc')
            h=hashlib.sha256(b'abc').hexdigest();lock=root/(h+'.lock')
            r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[])
            code='from filelock import FileLock;import sys;lock=FileLock(sys.argv[1]);lock.acquire();print("locked",flush=True);sys.stdin.read()'
            child=subprocess.Popen([sys.executable,'-B','-c',code,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'locked')
                with self.assertRaises(FileExistsError):fetch_resource(r,root,local_file=source)
                child.kill();child.wait(timeout=10)
                try:receipt=fetch_resource(r,root,local_file=source)
                except FileExistsError:self.fail('process death did not release the download lock')
                self.assertEqual(Path(receipt['path']).read_bytes(),b'abc')
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=10)
                child.stdin.close();child.stdout.close();child.stderr.close()

    def test_resume_interrupted_local_import(self):
        from riblet.resource_fetch import fetch_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_bytes(b'abc')
            h=hashlib.sha256(b'abc').hexdigest();(root/(h+'.importing')).write_bytes(b'a')
            r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[])
            self.assertEqual(Path(fetch_resource(r,root,local_file=source)['path']).read_bytes(),b'abc')

    def test_unpack_exact_payload_and_no_overwrite(self):
        import tarfile,io
        from riblet.resource_fetch import unpack_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'a.tar'
            with tarfile.open(archive,'w') as t:
                info=tarfile.TarInfo('sub/x');info.size=3;t.addfile(info,io.BytesIO(b'abc'))
            members=[dict(path='sub/x',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())]
            unpack_resource(archive,members,root/'out')
            self.assertEqual((root/'out/sub/x').read_bytes(),b'abc')
            with self.assertRaises(FileExistsError):unpack_resource(archive,members,root/'out')

    def test_unpack_bad_hash_and_symlink_never_admitted(self):
        import tarfile,io
        from riblet.resource_fetch import unpack_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for kind in ('hash','link','duplicate'):
                archive=root/(kind+'.tar')
                with tarfile.open(archive,'w') as t:
                    info=tarfile.TarInfo('x');info.size=3
                    if kind=='link':info.type=tarfile.SYMTYPE;info.linkname='../escape';info.size=0;t.addfile(info)
                    else:
                        t.addfile(info,io.BytesIO(b'abc'))
                        if kind=='duplicate':t.addfile(info,io.BytesIO(b'abc'))
                members=[dict(path='x',bytes=3,sha256='0'*64)]
                with self.assertRaises(ValueError):unpack_resource(archive,members,root/kind)
                self.assertFalse((root/kind).exists())

    def test_network_resume_and_ignored_range(self):
        import threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from riblet.resource_fetch import fetch_resource
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                seen.append(self.headers.get('Range'))
                if self.path=='/range' and self.headers.get('Range'):
                    self.send_response(206);self.send_header('Content-Range','bytes 1-2/3');body=b'bc'
                else:self.send_response(200);body=b'abc'
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for path in ('range','ignore'):
                with tempfile.TemporaryDirectory() as d:
                    h=hashlib.sha256(b'abc').hexdigest();(Path(d)/(h+'.part')).write_bytes(b'a')
                    r=dict(resource_id='x',version='v1',bytes=3,sha256=h,urls=[f'http://127.0.0.1:{server.server_port}/{path}'])
                    out=fetch_resource(r,d)
                    self.assertEqual(Path(out['path']).read_bytes(),b'abc')
            self.assertEqual(seen,['bytes=1-','bytes=1-'])
        finally:server.shutdown();server.server_close();thread.join()

    def test_safe_unpack_rejects_traversal(self):
        import tarfile,io
        from riblet.resource_fetch import unpack_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'a.tar'
            with tarfile.open(archive,'w') as t:
                info=tarfile.TarInfo('../escape');info.size=3;t.addfile(info,io.BytesIO(b'abc'))
            with self.assertRaises(ValueError):unpack_resource(archive,[],root/'out')
            self.assertFalse((root/'escape').exists())

    def test_import_cache_and_corruption(self):
        from riblet.resource_fetch import fetch_resource
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_bytes(b'abc')
            r=dict(resource_id='x',version='v1',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest(),urls=[])
            receipt=fetch_resource(r,root/'cache',local_file=source)
            self.assertEqual(Path(receipt['path']).read_bytes(),b'abc')
            source.unlink()
            self.assertEqual(fetch_resource(r,root/'cache',offline=True)['status'],'cached')
            Path(receipt['path']).write_bytes(b'bad')
            with self.assertRaises(ValueError):fetch_resource(r,root/'cache',offline=True)

    def test_wrong_import_not_admitted(self):
        from riblet.resource_fetch import fetch_resource
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input';source.write_bytes(b'bad')
            r=dict(resource_id='x',version='v1',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest(),urls=[])
            with self.assertRaises(ValueError):fetch_resource(r,Path(d)/'cache',local_file=source)
            self.assertFalse((Path(d)/'cache'/r['sha256']).exists())
