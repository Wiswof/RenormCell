"""Optional publisher download, separate from the small inference asset bundle."""
import argparse
import json
from pathlib import Path
from .assets import DATA,download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--file-index',type=int,help='Required to download; omit to list files and sizes')
    args=p.parse_args()
    manifest=DATA/'raw_downloads.json'
    rows=[r for r in json.loads(manifest.read_text()) if r['study']==args.study]
    if not rows: raise ValueError('Unknown study; see data/sources.csv')
    if args.file_index is None:
        for i,r in enumerate(rows): print(i,r.get('filename',''),r.get('bytes','unknown size'),r['url'])
        return
    if not 0<=args.file_index<len(rows): raise ValueError('Invalid publisher file index')
    row=rows[args.file_index]
    from urllib.parse import urlparse,unquote
    name=row.get('filename') or Path(unquote(urlparse(row['url']).path)).name
    if not name or name in ('.','..') or Path(name).name!=name: raise ValueError('Unsafe publisher filename')
    dest=Path(args.output)/args.study/name
    download(row['url'],dest,row.get('sha256'))
    from .io import digest,write_json
    write_json(dest.with_name(dest.name+'.receipt.json'),dict(url=row['url'],sha256=digest(dest),
        bytes=dest.stat().st_size,publisher_sha256_checked=bool(row.get('sha256'))))
    print('DOWNLOADED',dest,'Apply author labels/QC; do not pass raw deposits directly to training.')


if __name__=='__main__':main()
