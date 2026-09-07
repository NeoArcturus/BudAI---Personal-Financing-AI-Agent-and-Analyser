import re
with open("app/(protected)/connections/ConnectionWidgets.tsx", "r") as f:
    content = f.read()

content = content.replace('key={b.bucket_id}', 'key={b.id || b.bucket_id || idx}')
content = content.replace('filteredBuckets.map((b: any)', 'filteredBuckets.map((b: any, idx: number)')

with open("app/(protected)/connections/ConnectionWidgets.tsx", "w") as f:
    f.write(content)
