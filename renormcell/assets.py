"""Download pinned public release assets, safely extract, verify every payload file."""
import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request
import sys
from .io import digest

DATA = Path(__file__).resolve().parents[1]/'data'
if not (DATA/'assets.json').is_file(): DATA = Path(sys.prefix)/'share/renormcell'
MANIFEST = DATA/'assets.json'


def download(url, path, expected):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and expected and digest(path)==expected: return
    part = path.with_name(path.name+'.part')
    start = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent':'RenormCell/1.0'}
    if start: headers['Range'] = f'bytes={start}-'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
        resume = start and response.status==206
        with part.open('ab' if resume else 'wb') as f:
            while block:=response.read(8*1024**2): f.write(block)
    if expected and digest(part)!=expected: raise ValueError(f'Checksum failed: {path.name}; partial retained')
    part.replace(path)


def verify(root, files):
    for relative,info in files.items():
        p = root/relative
        if not p.is_file() or p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
            raise ValueError(f'Asset verification failed: {relative}')


def fetch(root):
    root=Path(root); manifest=json.loads(MANIFEST.read_text())
    ready=root/'FILES.json'
    if ready.exists():
        files=json.loads(ready.read_text())
        if digest(ready)!=manifest['files_sha256']: raise ValueError('Different asset release; use a new directory')
        try:
            verify(root,files); print('ASSETS_VERIFIED',root,flush=True); return
        except ValueError:
            print('RESUMING incomplete/corrupt asset installation',flush=True)
    for item in manifest['archives']:
        path=root/'.downloads'/item['name']
        print('DOWNLOAD',item['name'],flush=True)
        download(item['url'],path,item['sha256'])
        with tarfile.open(path,'r:gz') as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe release archive member')
                if name.parts[0] not in ('atlas','protected','gene','weights','FILES.json'):
                    raise ValueError('Unexpected asset namespace')
                archive.extract(member,root,filter='data')
    if digest(ready)!=manifest['files_sha256']: raise ValueError('File manifest checksum mismatch')
    verify(root,json.loads(ready.read_text())); print('ASSETS_VERIFIED',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='.assets')
    args=p.parse_args();fetch(args.output)


if __name__=='__main__':main()
