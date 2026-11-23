"""Reproduce the colleague-run OCR EDA. Python 3.11+ and matplotlib required.
Run python analysis.py, or open OCR_EDA.ipynb from this directory.
Only input/ocr_labels.txt supplies truth text and marking categories.
"""
from pathlib import Path
from collections import Counter
import csv, json, re, statistics, math, html, base64, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
INPUT=BASE/'input';OUT=BASE/'results';OUT.mkdir(exist_ok=True)
PROFILES=['baseline','large'];LABEL={'baseline':'EasyOCR','large':'Florence-2'}
COLORS=['#267991','#dd7545']
def norm(s):return re.sub(r'\s+','',s).upper()
def stem(s):return Path(s.replace('\\','/')).stem
def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]
def percentile(v,q):
    v=sorted(v);p=(len(v)-1)*q;l=math.floor(p);h=math.ceil(p)
    return v[l]+(p-l)*(v[h]-v[l])
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
with (INPUT/'ocr_labels.txt').open(encoding='utf-8-sig',newline='') as f:
    label_rows=list(csv.DictReader(f,delimiter=';'))
truth={}
for r in label_rows:
    key=stem(r['File Name'])
    assert key not in truth,'Duplicate reference: '+key
    assert r['Category'].strip() and norm(r['Label (Human)'])
    truth[key]={'truth':r['Label (Human)'],'category':r['Category'].strip()}
data=json.loads((INPUT/'ocr_all_results.json').read_text())
images={stem(i['image']):i for i in data['images']}
assert len(images)==len(data['images'])==data['image_count']==134
assert set(images)==set(truth),'Labels and images must match exactly'
rows=[]
for id,im in images.items():
    assert len(im['models'])==2 and {m['profile'] for m in im['models']}==set(PROFILES)
    t=truth[id]['truth']
    for m in im['models']:
        pred=m.get('text') or ''
        rd,nd=distance(t,pred),distance(norm(t),norm(pred))
        rows.append(dict(image=id,category=truth[id]['category'],profile=m['profile'],label=m['label'],
            truth_text=t,ocr_text=pred,status=m['status'],error=m.get('error'),
            raw_reference_length=len(t),normalized_reference_length=len(norm(t)),raw_edit_distance=rd,
            normalized_edit_distance=nd,raw_cer=rd/len(t),normalized_cer=nd/len(norm(t)),
            exact_match_raw=int(t==pred),exact_match_normalized=int(norm(t)==norm(pred)),
            analysis_seconds=m['processing_seconds'],total_seconds=m['total_seconds'],
            startup_seconds=m['startup_seconds'],model_reused=m['model_reused'],
            width=im['image_size'][0],height=im['image_size'][1]))
assert len(rows)==268 and len({(r['image'],r['profile']) for r in rows})==268
def summarize(rr):
    n=len(rr);k=sum(r['exact_match_normalized'] for r in rr);lo,hi=wilson(k,n)
    times=[r['analysis_seconds'] for r in rr]
    return dict(samples=n,normalized_exact_count=k,normalized_exact_match=k/n,ci_low=lo,ci_high=hi,
        raw_exact_count=sum(r['exact_match_raw'] for r in rr),raw_exact_match=statistics.mean(r['exact_match_raw'] for r in rr),
        mean_normalized_cer=statistics.mean(r['normalized_cer'] for r in rr),
        pooled_normalized_cer=sum(r['normalized_edit_distance'] for r in rr)/sum(r['normalized_reference_length'] for r in rr),
        median_normalized_cer=statistics.median(r['normalized_cer'] for r in rr),mean_raw_cer=statistics.mean(r['raw_cer'] for r in rr),
        mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),p95_seconds=percentile(times,.95),
        mean_total_seconds=statistics.mean(r['total_seconds'] for r in rr),successful=sum(r['status']=='ok' for r in rr),
        blank_outputs=sum(not norm(r['ocr_text']) for r in rr),cer_over_100=sum(r['normalized_cer']>1 for r in rr))
overall={p:summarize([r for r in rows if r['profile']==p]) for p in PROFILES}
cats=sorted({r['category'] for r in rows})
categories={(c,p):summarize([r for r in rows if r['category']==c and r['profile']==p]) for c in cats for p in PROFILES}
paired={p:{r['image']:r for r in rows if r['profile']==p} for p in PROFILES}
outcomes=Counter((paired['baseline'][i]['exact_match_normalized'],paired['large'][i]['exact_match_normalized']) for i in images)
agree=[i for i in images if norm(paired['baseline'][i]['ocr_text'])==norm(paired['large'][i]['ocr_text'])]
agree_correct=sum(paired['baseline'][i]['exact_match_normalized'] for i in agree)
lower_cer=Counter('Florence-2 lower' if paired['large'][i]['normalized_cer']<paired['baseline'][i]['normalized_cer'] else
    'EasyOCR lower' if paired['large'][i]['normalized_cer']>paired['baseline'][i]['normalized_cer'] else 'Tie' for i in images)
def writecsv(name,rr):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
writecsv('ocr_evaluation_detail.csv',rows)
writecsv('ocr_evaluation_summary.csv',[dict(profile=p,**overall[p]) for p in PROFILES])
writecsv('marking_type_summary.csv',[dict(category=c,profile=p,**categories[c,p]) for c in cats for p in PROFILES])
writecsv('paired_image_results.csv',[dict(image=i,category=truth[i]['category'],truth_text=truth[i]['truth'],
    easyocr_text=paired['baseline'][i]['ocr_text'],florence_text=paired['large'][i]['ocr_text'],
    easyocr_exact=paired['baseline'][i]['exact_match_normalized'],florence_exact=paired['large'][i]['exact_match_normalized'],
    easyocr_cer=paired['baseline'][i]['normalized_cer'],florence_cer=paired['large'][i]['normalized_cer']) for i in images])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
def save(fig,name):fig.savefig(OUT/name,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4.1))
for j,p in enumerate(PROFILES):
    s=overall[p];v=s['normalized_exact_match']*100
    ax[0].bar(j,v,color=COLORS[j],width=.55)
    ax[0].errorbar(j,v,yerr=[[v-100*s['ci_low']],[100*s['ci_high']-v]],fmt='none',ecolor='#243743',capsize=4)
    ax[0].text(j,100*s['ci_high']+2,f"{s['normalized_exact_count']}/134",ha='center')
    ax[1].bar(j,100*s['mean_normalized_cer'],color=COLORS[j],width=.55)
    ax[1].text(j,100*s['mean_normalized_cer']+1,f"{s['mean_normalized_cer']:.1%}",ha='center')
for a in ax:a.set_xticks([0,1],['EasyOCR','Florence-2']);a.set_ylabel('Percent');a.set_ylim(0,85)
ax[0].set_title('Normalized exact transcription match');ax[1].set_title('Mean normalized character error rate')
fig.text(.01,-.03,'134 paired images, scored using the newly supplied label file. Error bars: 95% Wilson intervals.',fontsize=9)
fig.tight_layout();save(fig,'overall.png')
fig,ax=plt.subplots(1,2,figsize=(11,4.1))
for j,p in enumerate(PROFILES):
    x=[i+(j-.5)*.34 for i in range(len(cats))]
    ax[0].bar(x,[100*categories[c,p]['normalized_exact_match'] for c in cats],width=.32,color=COLORS[j],label=LABEL[p])
    ax[1].bar(x,[100*categories[c,p]['mean_normalized_cer'] for c in cats],width=.32,color=COLORS[j])
for a in ax:a.set_xticks(range(len(cats)),[f"{c}\n(n={categories[c,'baseline']['samples']})" for c in cats]);a.set_ylabel('Percent');a.set_ylim(0,85)
ax[0].legend(frameon=False);ax[0].set_title('Normalized exact match by marking type');ax[1].set_title('Mean normalized CER by marking type')
fig.tight_layout();save(fig,'categories.png')
fig,ax=plt.subplots(1,2,figsize=(11,4.1))
for a,field,title,ylabel in [(ax[0],'normalized_cer','Distribution of character errors','Normalized CER (%)'),(ax[1],'analysis_seconds','Processing time excludes startup','Seconds per image')]:
    box=a.boxplot([[r[field]*(100 if field=='normalized_cer' else 1) for r in rows if r['profile']==p] for p in PROFILES],tick_labels=['EasyOCR','Florence-2'],patch_artist=True)
    for patch,color in zip(box['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.7)
    a.set_title(title);a.set_ylabel(ylabel)
fig.tight_layout();save(fig,'distributions.png')

def esc(s):return html.escape(str(s))
def table(head,data):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table></div>'
def picture(name,alt):return '<img alt="'+esc(alt)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
def summary_table(ss):return table(['Model','Normalized exact','95% interval','Mean CER','Pooled CER','Mean / median / P95 seconds'],[
    [LABEL[p],f"{s['normalized_exact_count']}/{s['samples']} ({s['normalized_exact_match']:.1%})",f"{s['ci_low']:.1%}–{s['ci_high']:.1%}",f"{s['mean_normalized_cer']:.1%}",f"{s['pooled_normalized_cer']:.1%}",f"{s['mean_seconds']:.2f} / {s['median_seconds']:.2f} / {s['p95_seconds']:.2f}"] for p,s in ss.items()])
category_table=table(['Category','Images','EasyOCR exact','Florence-2 exact','EasyOCR mean CER','Florence-2 mean CER'],[
    [esc(c),categories[c,'baseline']['samples'],f"{categories[c,'baseline']['normalized_exact_count']}/{categories[c,'baseline']['samples']} ({categories[c,'baseline']['normalized_exact_match']:.1%})",f"{categories[c,'large']['normalized_exact_count']}/{categories[c,'large']['samples']} ({categories[c,'large']['normalized_exact_match']:.1%})",f"{categories[c,'baseline']['mean_normalized_cer']:.1%}",f"{categories[c,'large']['mean_normalized_cer']:.1%}"] for c in cats])
worst=table(['Image','Category','Human reference','Florence-2 output','Normalized CER'],[
    [esc(r['image']),esc(r['category']),esc(r['truth_text']),'<pre>'+esc(r['ocr_text'])+'</pre>',f"{r['normalized_cer']:.1%}"] for r in sorted(paired['large'].values(),key=lambda r:r['normalized_cer'],reverse=True)[:6]])
near=[r for r in paired['large'].values() if 0<r['normalized_cer']<=.15]
near_table=table(['Image','Human reference','Florence-2 output','Normalized CER'],[
    [esc(r['image']),esc(r['truth_text']),'<pre>'+esc(r['ocr_text'])+'</pre>',f"{r['normalized_cer']:.1%}"] for r in sorted(near,key=lambda r:r['normalized_cer'])[:5]])
b,l=overall['baseline'],overall['large']
trailing=sum(t['truth']!=t['truth'].strip() for t in truth.values())
report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OCR EDA: colleague run, 134 labeled images</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#eef2f4;color:#23333e;margin:0}}main{{max-width:1120px;margin:auto;padding:44px;background:white}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:38px;font-size:25px}}.eyebrow{{color:#267991;font-weight:700;letter-spacing:1px;font-size:13px}}.lead{{font-size:20px}}.note{{background:#fff4e9;border-left:4px solid #dd7545;padding:16px 20px}}.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:25px 0}}.card{{flex:1;min-width:180px;background:#edf5f7;padding:18px}}.num{{font-size:30px;font-weight:750;display:block;color:#176078}}img{{width:100%;height:auto;margin:15px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #dde5e9;text-align:left;vertical-align:top}}th{{background:#edf5f7}}.table{{overflow-x:auto;margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;min-width:90px;margin:0}}code{{overflow-wrap:anywhere;font-size:12px}}li{{margin:8px 0}}.small{{font-size:13px;color:#53646f}}@media print{{main{{padding:10px}}h2{{break-after:avoid}}img,tr{{break-inside:avoid}}}}</style></head><body><main>
<div class="eyebrow">IMAGE OCR · COLLEAGUE RUN · 23 NOVEMBER 2026</div>
<h1>OCR evaluation of 134 images using the new human labels</h1>
<p class="lead">Florence-2 matched {l['normalized_exact_count']} normalized reference transcriptions ({l['normalized_exact_match']:.1%}); EasyOCR matched {b['normalized_exact_count']} ({b['normalized_exact_match']:.1%}). Florence-2 had a lower mean character error rate and required more processing time.</p>
<div class="cards"><div class="card"><span class="num">134 / 134</span>Images matched to new human labels</div><div class="card"><span class="num">{100*(l['normalized_exact_match']-b['normalized_exact_match']):.1f} pp</span>Florence-2 exact-match advantage</div><div class="card"><span class="num">{l['mean_seconds']/b['mean_seconds']:.1f}×</span>Mean processing-time ratio in this run</div></div>
<div class="note"><b>New image numbering and labels.</b> This export reuses earlier filenames for different pictures. This report uses only the newly supplied <code>ocr_labels.txt</code> for reference text and categories. Earlier crop-specific labels and category assignments are not applied. References describe the complete target transcription per image; many include surrounding text or multiple lines. These results do not yet measure isolated whole-ID or assembly-slot correctness.</div>
<h2>1 · Coverage and data quality</h2>
<p>The export has 134 unique images and 268 unique model/image records. Every image has exactly one EasyOCR and one Florence-2 result, and every result has status <code>ok</code>. All 134 images match one unique label row; there are no missing, duplicate, extra, or empty references. The new human labels classify 65 images as Engrave and 69 as Laser Mark. There are no separately labeled Stamp, Cast, hand-scribed, or chemically etched images in this reference file.</p>
<p>The semicolon-delimited label file is parsed as UTF-8 text. Identifiers, punctuation, leading zeros, and original whitespace are preserved. One label (<code>ocr_test_025</code>) has a trailing space; it is retained in raw scoring and ignored by normalized scoring. Successful execution does not establish correct recognition.</p>
<h2>2 · Overall model comparison</h2>{picture('overall.png','Normalized exact transcription match and mean CER on 134 labeled images')}{summary_table(overall)}
<p>Raw exact-match rates are {b['raw_exact_count']}/134 ({b['raw_exact_match']:.1%}) for EasyOCR and {l['raw_exact_count']}/134 ({l['raw_exact_match']:.1%}) for Florence-2. Raw mean CER is {b['mean_raw_cer']:.1%} versus {l['mean_raw_cer']:.1%}. These raw measures penalize formatting differences such as OCR line breaks versus spaces in human labels, so normalized results are the main comparison.</p>
<p>Mean processing times are {b['mean_seconds']:.2f} and {l['mean_seconds']:.2f} seconds per image; mean total times including recorded startup are {b['mean_total_seconds']:.2f} and {l['mean_total_seconds']:.2f} seconds. These values describe this supplied run. Hardware specifications and repeated-run variation are not included, so timing is not attributed to a particular processor or directly compared with earlier runs.</p>
<h2>3 · Marking-type comparison</h2>{picture('categories.png','Exact match and CER by the two newly labeled marking categories')}{category_table}
<p>The new category labels are authoritative for this report. Performance on engraved and laser-marked samples does not establish performance on other marking methods. Physical-part independence and production representativeness were not established by the exports.</p>
<h2>4 · Paired errors and model agreement</h2>
<p>Both models matched the reference on {outcomes[1,1]} images; only Florence-2 matched on {outcomes[0,1]}; only EasyOCR matched on {outcomes[1,0]}; neither matched on {outcomes[0,0]}. Florence-2 is therefore not uniformly better. Per-image CER comparisons: {esc(dict(lower_cer))}.</p>
<p>The models produced identical normalized text on {len(agree)}/134 images ({len(agree)/134:.1%}). Of those agreements, {agree_correct} matched the reference and {len(agree)-agree_correct} did not. Agreement is a diagnostic, not proof of correctness or a validated automatic-accept policy.</p>
<h2>5 · Error and processing-time distributions</h2>{picture('distributions.png','Distributions of normalized CER and processing time')}
<p>Blank outputs: EasyOCR {b['blank_outputs']}, Florence-2 {l['blank_outputs']}. Normalized CER above 100%: EasyOCR {b['cer_over_100']}, Florence-2 {l['cer_over_100']}. CER can exceed 100% when inserted characters exceed reference length; these values are retained without capping. A blank prediction is not evidence that an image contains no marking. No execution failures or timeouts are recorded in this run.</p>
<h2>6 · Examples for manual review</h2><p>These are the six highest normalized-CER Florence-2 cases. Review the original image and label before attributing a cause to glare, low contrast, detection, reading order, or transcription error.</p>{worst}
<p>Small errors can still invalidate an ID. The following near-match cases have nonzero normalized CER at or below 15% and therefore fail exact match.</p>{near_table}
<h2>7 · Method and validation</h2>
<ul><li><b>Exact match:</b> the entire predicted transcription must equal the human reference. One incorrect character fails exact match.</li>
<li><b>Normalization:</b> remove all whitespace and uppercase both strings. Preserve punctuation, leading zeros, and character distinctions such as O/0. Reading-order differences are not corrected. No model output is used to revise truth.</li>
<li><b>CER:</b> Levenshtein character edit distance (insertions, deletions, substitutions) divided by reference character count. AB123456 → AB12345G is 12.5% CER and an incorrect complete ID.</li>
<li><b>Mean CER:</b> average image-level CER, giving every image equal weight. <b>Pooled CER:</b> total character edits divided by total reference characters, giving longer references more weight. Both are reported.</li>
<li><b>Confidence intervals:</b> 95% Wilson binomial intervals for exact match assume independent image observations. They do not correct for repeated parts, selection bias, or reference-label errors.</li>
<li><b>Timing:</b> processing excludes model startup, while total time includes the recorded setup overhead. P95 is calculated by linear interpolation between ordered timings. No inference was rerun.</li>
<li><b>Audit:</b> image_count=134, unique label keys=134, unique image keys=134, image-model rows=268. Exact one-to-one label joins and profile coverage are checked by executable assertions. The analysis uses current predictions and the new labels exclusively. Source hashes are in <code>audit.json</code>.</li></ul>
<h2>8 · Interpretation and next experiments</h2>
<ol><li>Use these results to compare the two workflows on this labeled set. Earlier 140- and 71-image scores use different populations and references; differences do not establish a model improvement.</li>
<li>For the intended part-ID task, label target ID regions separately from surrounding text. For assembly inspection, retain assembly identity, photographed end, and physical slot. Score exact IDs and whole-assembly decisions in addition to CER.</li>
<li>Review near matches and high-error cases against original images. Keep meaningful punctuation and leading zeros. Normalize only according to the ID specification.</li>
<li>Run RapidOCR on the same 134 images with the same labels and reporting policy. This export contains only EasyOCR and Florence-2.</li>
<li>Test capture lighting and region cropping on development data, then evaluate on held-out physical parts. Add any production marking types absent here.</li>
<li>Before automatic acceptance, measure wrong accepts, automatic-accept accuracy, and manual-review rate on untouched evaluation data. OCR confidence and agreement are not substitutes for validated decision performance.</li></ol>
<h2>9 · Team-ready takeaway</h2>
<p>“On 134 images scored against newly supplied human labels, Florence-2 achieved {l['normalized_exact_match']:.1%} normalized exact-transcription accuracy versus {b['normalized_exact_match']:.1%} for EasyOCR. Mean normalized character error rate was {l['mean_normalized_cer']:.1%} versus {b['mean_normalized_cer']:.1%}. Mean processing time was {l['mean_seconds']:.2f} versus {b['mean_seconds']:.2f} seconds per image. These exploratory results cover engraved and laser-marked samples; target-ID and assembly-placement accuracy require separate evaluation.”</p>
<h2>Sources and reproduction</h2>
<p>Inputs: the uploaded <code>ocr_all_results(1).json</code>, copied without content changes to <code>input/ocr_all_results.json</code>, and <code>input/ocr_labels.txt</code>. No earlier ground-truth file or crop override is used. Original bytes are preserved.</p>
<p>Open <code>OCR_EDA.ipynb</code> in VS Code or Jupyter, or run <code>python analysis.py</code> with matplotlib installed. The package contains this report, all three plots, the executed notebook, the analysis script, raw inputs, scored detail, overall and category summaries, paired predictions, and a source/validation audit.</p>
<p class="small">Analysis uses supplied exports and human labels. Original images were not attached and reference transcription was not independently verified.</p></main></body></html>'''
(OUT/'OCR_EDA_Report.html').write_text(report,encoding='utf-8')
audit=dict(images=len(images),labels=len(truth),model_records=len(rows),label_source='ocr_labels.txt only',join='exact filename stem',
    categories=Counter(t['category'] for t in truth.values()),trailing_whitespace_labels=trailing,
    overall=overall,paired_outcomes={str(k):v for k,v in outcomes.items()},normalized_agreements=len(agree),correct_agreements=agree_correct,
    validation='passed: exact label coverage, unique images/labels/model records, two profiles per image, nonempty reference text',
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUT.iterdir())})
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
