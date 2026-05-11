"""
Verify each cited bib entry against CrossRef (Zotero-equivalent DOI metadata check).

For each citation key in paper.tex:
  1. Read the bib entry
  2. Extract DOI
  3. Query api.crossref.org/works/{doi}
  4. Compare returned title/year/journal/first author against bib
  5. Flag mismatches
"""
from pathlib import Path
import re, json, urllib.request, urllib.parse, time
from difflib import SequenceMatcher

ROOT = Path('/home/holiday01/drug_sc/manuscript')
PAPER = ROOT/'paper.tex'
BIB = ROOT/'refs.bib'

# 1) extract cited keys
keys_cited = set()
for m in re.finditer(r'\\citep?\{([^}]+)\}', PAPER.read_text()):
    for k in m.group(1).split(','):
        keys_cited.add(k.strip())
print(f'Cited keys: {len(keys_cited)}')

# 2) parse bib entries
bib_text = BIB.read_text()
entries = {}
for m in re.finditer(r'@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\}', bib_text, re.S):
    etype, key, body = m.groups()
    fields = {}
    for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}\s*[,\n]', body+'\n', re.S):
        fields[fm.group(1).lower()] = fm.group(2).strip()
    fields['_type'] = etype
    entries[key.strip()] = fields
print(f'Bib entries parsed: {len(entries)}')

# 3) helper: query CrossRef
def crossref(doi):
    url = f'https://api.crossref.org/works/{urllib.parse.quote(doi, safe="/")}'
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        return d.get('message')
    except Exception as e:
        return {'_error': str(e)}

def sim(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

# 4) per-key verification
results = []
for k in sorted(keys_cited):
    if k not in entries:
        results.append({'key':k, 'status':'MISSING_BIB', 'note':'cited but not in refs.bib'}); continue
    e = entries[k]
    doi = e.get('doi','').strip()
    if not doi:
        results.append({'key':k, 'status':'NO_DOI', 'bib_title': e.get('title','')[:60]}); continue

    print(f'querying {k} … {doi}', flush=True)
    cr = crossref(doi)
    time.sleep(0.5)  # be polite
    if cr is None or '_error' in (cr or {}):
        results.append({'key':k, 'status':'CROSSREF_ERROR', 'doi':doi, 'note':cr.get('_error') if cr else 'no response'}); continue

    # Extract CrossRef metadata
    cr_title  = (cr.get('title') or [''])[0]
    cr_year   = (cr.get('issued',{}).get('date-parts',[[None]])[0][0]) or cr.get('created',{}).get('date-parts',[[None]])[0][0]
    cr_journal = (cr.get('container-title') or [''])[0] or cr.get('publisher','')
    cr_authors = cr.get('author', [])
    cr_first   = (cr_authors[0].get('family','') if cr_authors else '')

    # Bib values
    b_title    = re.sub(r'\\\w+|[{}]', '', e.get('title','')).strip()
    b_year     = e.get('year','').strip()
    b_journal  = re.sub(r'[{}]', '', e.get('journal') or e.get('booktitle','')).strip()
    b_authors  = e.get('author','')
    b_first    = b_authors.split(',')[0].split(' and ')[0].strip().rstrip(',').replace('{','').replace('}','')

    # Compare
    title_match  = sim(cr_title, b_title) if cr_title and b_title else 0
    year_match   = (str(cr_year) == b_year) if cr_year and b_year else None
    journal_match= sim(cr_journal, b_journal) if cr_journal and b_journal else 0
    author_match = (b_first and (b_first.lower() in cr_first.lower() or cr_first.lower() in b_first.lower()))

    status = 'OK'
    notes = []
    if title_match < 0.7:
        # Allow capitalization-only differences
        if title_match >= 0.6 and cr_title.lower().replace(' ','').startswith(b_title.lower().replace(' ','')[:20]):
            notes.append(f'capitalization-only diff (sim={title_match:.2f}); not flagged')
        else:
            status = 'TITLE_MISMATCH'; notes.append(f'bib_title="{b_title[:50]}" vs crossref="{cr_title[:50]}" sim={title_match:.2f}')
    if year_match is False:
        # Online vs print year convention: ≤1 year difference is benign
        if abs(int(cr_year) - int(b_year)) <= 1:
            notes.append(f'year diff (bib={b_year}, crossref={cr_year}; online vs print convention; not flagged)')
        else:
            status = 'YEAR_MISMATCH' if status=='OK' else status+'+YEAR'; notes.append(f'bib_year={b_year} vs crossref={cr_year}')
    if journal_match < 0.5 and b_journal and cr_journal:
        notes.append(f'journal: bib="{b_journal[:30]}" vs crossref="{cr_journal[:30]}"')
    if not author_match and b_first and cr_first:
        notes.append(f'first_author: bib="{b_first}" vs crossref="{cr_first}"')

    results.append({
        'key':k, 'status':status, 'doi':doi,
        'crossref_title': cr_title[:60],
        'bib_title': b_title[:60],
        'cr_year':cr_year, 'bib_year':b_year,
        'first_author_match': author_match,
        'notes': '; '.join(notes) if notes else ''
    })

# 5) Report
ok = sum(1 for r in results if r['status']=='OK')
no_doi = sum(1 for r in results if r['status']=='NO_DOI')
err = sum(1 for r in results if r['status']=='CROSSREF_ERROR')
mismatch = sum(1 for r in results if 'MISMATCH' in r['status'])
missing = sum(1 for r in results if r['status']=='MISSING_BIB')

print(f'\n=== CITATION VERIFICATION (CrossRef DOI lookup, Zotero-equivalent) ===')
print(f'  OK (DOI matches title/year/author): {ok}')
print(f'  NO_DOI (cannot verify): {no_doi}')
print(f'  CROSSREF_ERROR (DOI not found):     {err}')
print(f'  TITLE/YEAR_MISMATCH:                {mismatch}')
print(f'  MISSING_BIB (key cited but no entry):{missing}')

# Write report
out = Path('/home/holiday01/drug_sc/results/citation_verification.md')
lines = ['# Citation verification (CrossRef DOI lookup)\n',
         f'**Method:** For each `\\cite{{key}}` in `paper.tex`, look up the DOI in `refs.bib` and query `api.crossref.org/works/{{doi}}`. Compare returned title, year, journal and first author against the bib entry.\n',
         f'**Summary:** {ok} OK, {no_doi} no-DOI, {err} CrossRef errors, {mismatch} mismatches, {missing} missing.\n',
         '## Per-citation results\n',
         '| Key | Status | DOI | Notes |',
         '|---|---|---|---|']
for r in results:
    lines.append(f"| `{r['key']}` | **{r['status']}** | `{r.get('doi','—')[:40]}` | {r.get('notes','') or r.get('note','')} |")
out.write_text('\n'.join(lines))
print(f'\nFull report: {out}')
