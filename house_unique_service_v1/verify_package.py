from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent;m=json.loads((r/'MANIFEST.json').read_text());bad=[]
for item in m['files']:
 p=r/item['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:bad.append(item['path'])
print('OK: '+str(len(m['files']))+' files verified' if not bad else 'FAILED: '+', '.join(bad))
sys.exit(bool(bad))
