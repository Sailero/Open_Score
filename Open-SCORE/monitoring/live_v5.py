"""Read-only live reports alongside a frozen v5 run; never imports a trainer."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
from urllib.parse import quote

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

TASKS = {
    'T1': ('真实终局 rollout', ['T1_rollout'], '真实终局模拟能否发现并选择有效的动态编组？'),
    'T2': ('MCTS-DPW', ['T2_mcts_dpw'], '多事件树搜索是否带来额外决策收益？'),
    'T3': ('候选 PPO 与 AR-PPO', ['t3_candidate', 't3_autoregressive'], '候选选择和自回归完整编组能否通过 PPO 学到有效决策？'),
    'T4': ('ExIt 风格策略迭代', ['t4_exit'], '模拟教师的收益能否通过逐轮蒸馏迁移到学生？'),
    'T5': ('BRIDGE 分组 DQN', ['t5_bridge_grouping'], '固定目标与身份归属后，学习组内合并是否有效？'),
    'T6': ('配对价值学习', ['t6_bce', 't6_adv', 't6_adv_gated'], '配对优势监督及门控是否改善方案选择？'),
}
CONTROLS = ['rule', 'grand', 'singleton', 'frozen_b1', 'frozen_b3']
NAMES = dict(T1_rollout='真实 rollout', T2_mcts_dpw='MCTS-DPW', t3_candidate='候选 PPO',
    t3_autoregressive='AR-PPO', t4_exit='ExIt 学生', t5_bridge_grouping='BRIDGE 分组 DQN',
    t6_bce='BCE 价值', t6_adv='ADV 价值', t6_adv_gated='ADV-gated', rule='规则上层',
    grand='每目标大组', singleton='同归属单体组', frozen_b1='冻结 v4 B1', frozen_b3='冻结 v4 B3')
FINAL = dict.fromkeys(['T1_rollout', 'T2_mcts_dpw', 't3_candidate', 't3_autoregressive', 't4_exit'], 'latest')
FINAL.update(t5_bridge_grouping='final_construction', t6_bce='final_epoch', t6_adv='final_epoch', t6_adv_gated='final_epoch')
FINAL.update(dict.fromkeys(CONTROLS, 'frozen'))
METHOD_TASK = {m: t for t, (_, methods, _) in TASKS.items() for m in methods} | dict.fromkeys(CONTROLS, 'T1')
RATIOS = {0.5: '较易', 0.75: '中等', 1.0: '等人数'}
TZ = timezone(timedelta(hours=8))


def local_time(value=None):
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if value else datetime.now(TZ)
        return dt.astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return str(value)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {} if default is None else default


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content if isinstance(content, bytes) else content.encode('utf-8')
    if path.exists() and path.read_bytes() == data:
        return
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


class Inputs:
    """Incremental append-only JSONL reader. An unfinished line is never a loss."""
    def __init__(self):
        self.cache = {}
        self.warnings = set()

    def rows(self, path):
        path = Path(path)
        if not path.exists():
            return []
        stat = path.stat()
        old = self.cache.get(path)
        if old and old['size'] == stat.st_size and old['mtime'] == stat.st_mtime_ns:
            return old['rows']
        if not old or stat.st_size <= old['size']:
            old = dict(size=0, tail=b'', rows=[])
        with path.open('rb') as stream:
            stream.seek(old['size'])
            data = old['tail'] + stream.read()
            size = stream.tell()
        chunks = data.split(b'\n')
        for line in chunks[:-1]:
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode('utf-8-sig'))
                if isinstance(row, dict):
                    old['rows'].append(row)
            except (ValueError, UnicodeError):
                self.warnings.add(f'忽略一条格式损坏的完整记录：{path}')
        old.update(size=size, tail=chunks[-1], mtime=stat.st_mtime_ns)
        self.cache[path] = old
        return old['rows']


def valid_episodes(rows):
    unique = {}
    for row in rows:
        result = row.get('success_native', row.get('success'))
        if row.get('family_id') and result in (True, False, 0, 1) and result is not None and row.get('execution_status', 'ok') not in ('error', 'failed_execution'):
            unique[row['family_id']] = row
    return sorted(unique.values(), key=lambda r: r['family_id'])


def win(row):
    return int(bool(row.get('success_native', row.get('success'))))


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    # Roundoff at k=0/n can otherwise put the endpoint epsilon beyond p.
    return [min(p, max(0., mid-half)), max(p, min(1., mid+half))]


def paired(rows, baseline):
    from scipy.stats import beta
    lookup = {r['family_id']: r for r in baseline}
    pairs = [(win(r), win(lookup[r['family_id']])) for r in rows if r['family_id'] in lookup]
    n = len(pairs)
    if not n:
        return dict(n=0, difference=None, interval=[None, None])
    plus = sum(a > b for a, b in pairs)
    minus = sum(a < b for a, b in pairs)
    def cp(k):
        return [0. if k == 0 else float(beta.ppf(.0125, k, n-k+1)),
                1. if k == n else float(beta.ppf(.9875, k+1, n-k))]
    a, b = cp(plus), cp(minus)
    return dict(n=n, more=plus, fewer=minus, difference=(plus-minus)/n,
                interval=[a[0]-b[1], a[1]-b[0]])


def aggregate(rows, cells, per_cell):
    groups = []
    for red, blue in cells:
        subset = [r for r in rows if (r.get('red_count'), r.get('blue_count')) == (red, blue)]
        k, n = sum(map(win, subset)), len(subset)
        groups.append(dict(red=red, blue=blue, wins=k, n=n, rate=k/n if n else None,
                           interval=wilson(k, n), complete=n == per_cell))
    k, n = sum(map(win, rows)), len(rows)
    difficulty = []
    for ratio, name in RATIOS.items():
        selected = [g for g in groups if g['blue']/g['red'] == ratio]
        total = sum(g['n'] for g in selected)
        successes = sum(g['wins'] for g in selected)
        difficulty.append(dict(name=name, ratio=ratio, wins=successes, n=total,
            rate=successes/total if total else None))
    observed = [g['rate'] for g in groups if g['n']]
    return dict(wins=k, n=n, expected=len(cells)*per_cell, rate=k/n if n else None,
        macro=sum(observed)/len(observed) if observed else None, interval=wilson(k, n),
        complete=bool(cells) and n == len(cells)*per_cell and all(g['complete'] for g in groups),
        cells=groups, difficulty=difficulty)


def checkpoint_axis(checkpoint):
    for prefix, axis in [('step_', 'Physical steps'), ('construction_', 'Construction episodes'), ('round_', 'ExIt rounds')]:
        if re.fullmatch(prefix+r'\d+', checkpoint):
            return axis, int(checkpoint[len(prefix):])
    return 'Checkpoint', 0


def get_evaluations(run, reader):
    result = {}
    for path in run.glob('T*/evaluations/*/*/*'):
        index = path/'episodes.jsonl'
        if not index.exists():
            continue
        rows = valid_episodes(reader.rows(index))
        if rows:
            method = rows[0].get('method_id', path.parents[1].name)
            split = rows[0].get('split', path.parent.name)
            checkpoint = rows[0].get('checkpoint_id', path.name)
            result[method, split, checkpoint] = dict(rows=rows, source=index.relative_to(run).as_posix(),
                summary=read_json(path/'summary.json'))
    return result


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def collect(run, reader):
    config = read_json(run/'shared/config_resolved.json')
    if not config.get('cells'):
        raise ValueError('Missing resolved experiment config')
    cells = config['cells']
    evaluations = get_evaluations(run, reader)
    gate = read_json(run/'T6/gate_selection.json')
    for entry in gate.get('thresholds',[]):
        if entry['threshold'] == gate.get('chosen_threshold'):
            gate_method = entry.get('summary',{}).get('method_id')
            source = evaluations.get((gate_method,'validation','final_epoch'))
            if source:
                evaluations['t6_adv_gated','validation',f"gate_{entry['threshold']}"] = source
    baseline = evaluations.get(('rule', 'test', 'frozen'), {}).get('rows', [])
    baseline_complete = aggregate(baseline, cells, config['eval_per_cell'])['complete']
    tasks = {}
    methods = {}
    for task, (title, arms, question) in TASKS.items():
        progress = read_json(run/task/'progress.json')
        result = read_json(run/task/'task_result.json')
        error = read_json(run/task/'execution_error.json')
        state = '已完成' if result.get('complete') else '执行异常' if error else '进行中' if progress else '尚未启动'
        tasks[task] = dict(title=title, question=question, arms=arms, state=state, progress=progress,
                           error=error.get('error'), updated=progress.get('updated'), curves=[])
    for method, checkpoint in FINAL.items():
        task = METHOD_TASK[method]
        native = evaluations.get((method, 'test', checkpoint), {})
        rows = native.get('rows', [])
        score = aggregate(rows, cells, config['eval_per_cell'])
        comparison = paired(rows, baseline)
        effect = '等待完整配对测试'
        if method == 'rule':
            effect = '公共参照'
        elif score['complete'] and baseline_complete and comparison['n'] == score['expected']:
            effect = '本次单种子有正收益证据' if comparison['interval'][0] > 0 else (
                '未观察到正的胜率差' if comparison['difference'] <= 0 else '正差值尚不确定')
        if config.get('smoke'):
            effect = '仅工程短测'
        validation = []
        for (m, split, ck), item in evaluations.items():
            if m != method or split != 'validation':
                continue
            stats = aggregate(item['rows'], cells, config['validation_per_cell'])
            axis, x = checkpoint_axis(ck)
            validation.append(dict(checkpoint=ck, x=x, axis=axis, source=item['source'], **stats))
        validation.sort(key=lambda v: (v['x'], v['checkpoint']))
        training = [r for r in reader.rows(run/task/'training.jsonl') if
            (r.get('method_id') or ('t6_'+r['kind'] if task == 'T6' and r.get('kind') else
                                   't5_bridge_grouping' if task == 'T5' else task)) == method]
        offline = [dict(epoch=r['epoch'], **r['validation']) for r in training if r.get('validation')]
        methods[method] = dict(id=method, name=NAMES[method], task=task, checkpoint=checkpoint,
            test=score, paired=comparison, effect=effect, validation=validation, offline=offline,
            training=training, test_rows=rows, source=native.get('source'), plots=[],
            last_training_time=training[-1].get('timestamp_utc') if training else None)
    diagnostic_source = run/'diagnostics.csv'
    diagnostics = []
    if diagnostic_source.exists():
        with diagnostic_source.open(encoding='utf-8-sig', newline='') as stream:
            diagnostics = list(csv.DictReader(stream))
    teacher_pairs = []
    for rnd in range(1, config.get('t4', {}).get('rounds', 4)+1):
        teacher = evaluations.get((f't4_teacher_round_{rnd}', 'validation', f'round_{rnd}'), {}).get('rows', [])
        student = evaluations.get((f't4_student_subset_round_{rnd}', 'validation', f'round_{rnd}'), {}).get('rows', [])
        if not student:
            student = evaluations.get(('t4_exit', 'validation', f'round_{rnd}'), {}).get('rows', [])
        reference = {r['family_id']: r for r in student}
        matched = [(r, reference[r['family_id']]) for r in teacher if r['family_id'] in reference]
        if matched:
            teacher_pairs.append(dict(round=rnd, n=len(matched), teacher=sum(win(a) for a,b in matched),
                                      student=sum(win(b) for a,b in matched)))
    summary = read_json(run/'comparison_summary.json')
    # Only the frozen runner can certify that fixed diagnostics and serial latency are all finished.
    complete = (summary.get('complete', False) and all(t['state'] == '已完成' for t in tasks.values())
                and all(m['test']['complete'] for m in methods.values()))
    return dict(updated=local_time(), config=config, tasks=tasks, methods=methods,
        diagnostics=diagnostics, diagnostics_updated=local_time(datetime.fromtimestamp(diagnostic_source.stat().st_mtime, timezone.utc).isoformat()) if diagnostic_source.exists() else None,
        teacher_pairs=teacher_pairs, gate=gate,
        partition=read_json(run/'T5/pure_partition_diagnostic.json'),
        complete=bool(complete), warnings=sorted(reader.warnings),
        method_commit=read_json(run/'shared/budget_manifest.json').get('git_commit'))


def pct(value):
    return '尚无数据' if value is None else f'{value:.1%}'


def fraction(score):
    return f"{score['wins']}/{score['n']}（{pct(score['rate'])}）" if score['n'] else '尚无数据'


def validation_last(method):
    completed = [v for v in method['validation'] if v['complete']]
    if completed:
        v = completed[-1]
        return f"{v['checkpoint']}：{fraction(v)}"
    if method['offline']:
        v = method['offline'][-1]
        return f"留出 epoch {v['epoch']}：排序准确率 {pct(v.get('ranking_accuracy'))}，配对 MSE {v.get('advantage_mse', '未记录')}"
    if method['id'] in CONTROLS or method['id'] in ('T1_rollout','T2_mcts_dpw'):
        return '规划/冻结策略：查看独立测试与预算诊断'
    return '尚无完整在线验证'


def progress_text(task):
    p = task['progress']
    bits = [task['state'], p.get('phase', '')]
    for done, total, label in [('physical_steps','total_physical_steps','物理步'),
        ('completed_episodes','total_episodes','回合'),('completed_states','total_states','状态'),
        ('completed','total','当前阶段配额'),
        ('epoch','epochs','epoch'),('round','rounds','轮')]:
        if done in p:
            bits.append(f"{label} {p[done]}/{p.get(total,'?')}")
    if p.get('method_id') or p.get('method'):
        bits.append(str(p.get('method_id', p.get('method'))))
    if task.get('error'):
        bits.append(task['error'])
    return ' · '.join(b for b in bits if b)


def build_plots(method, output, cache):
    """One panel per metric/physical meaning; no interpolation of missing validation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = defaultdict(list)
    def add(panel, axis, label, x, y):
        if finite(x) and finite(y):
            series[panel, axis, label].append((x, y))
    for row in method['training']:
        if 'physical_steps' in row:
            x, axis = row['physical_steps'], 'Physical training steps'
        elif 'construction_episode' in row:
            x, axis = row['construction_episode'], 'Construction episodes'
        else:
            x, axis = row.get('epoch'), 'Epoch'
        phase = row.get('phase', 'train')
        if phase == 'distill':
            phase += f" round {row.get('round')}"
        for metric in ('loss','actor_loss','value_loss','td_loss'):
            add('Loss (not task success)', axis, phase+' '+metric, x, row.get(metric))
        add('Training native success (window)', axis, 'native win rate', x, row.get('success_window'))
        if 'return' in row:
            add('Paired potential difference (not native win rate)', axis, 'construction return', x, row['return'])
    for v in method['validation']:
        if not v['complete']:
            continue
        add('Validation native success', v['axis'], 'all configured cells', v['x'], v['rate'])
        for d in v['difficulty']:
            add('Validation native success', v['axis'], f"blue/red={d['ratio']:g}", v['x'], d['rate'])
    for row in method['offline']:
        for metric in ('brier', 'ece', 'advantage_mse', 'ranking_accuracy', 'empirical_candidate_regret'):
            add('Held-out '+metric+' (not native win rate)', 'Epoch', metric, row['epoch'], row.get(metric))
    # This is test coverage, not a learning curve; sort in a reproducible opening order.
    running = 0
    for i, row in enumerate(method['test_rows'], 1):
        running += win(row)
        add('Final-policy cumulative test (not learning)', 'Completed test openings (family order)', 'native win rate', i, running/i)
    serial = [{'panel':k[0], 'axis':k[1], 'label':k[2], 'points':v} for k,v in sorted(series.items())]
    key = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
    image_path = output/'curves.png'
    if cache.get(method['id']) == key and (image_path.exists() or not series):
        return serial
    atomic(output/'curves.json', json.dumps(serial, ensure_ascii=False, indent=2))
    if series:
        panels = sorted({(k[0], k[1]) for k in series})
        fig, axes = plt.subplots(math.ceil(len(panels)/2), 2, squeeze=False,
                                 figsize=(13, 3.7*math.ceil(len(panels)/2)))
        for ax, (panel, axis) in zip(axes.flat, panels):
            for (p, a, label), points in series.items():
                if (p, a) == (panel, axis):
                    ax.plot([x for x,y in points], [y for x,y in points],
                            marker='o' if len(points)<50 else None, markersize=3, linewidth=1.4, label=label)
            ax.set(title=panel, xlabel=axis)
            if 'native success' in panel or 'cumulative test' in panel or 'ranking_accuracy' in panel:
                ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(method['id'], fontsize=14)
        fig.tight_layout()
        import io
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120)
        plt.close(fig)
        atomic(image_path, buffer.getvalue())
    cache[method['id']] = key
    return serial


STYLE = """
:root{color-scheme:light;--ink:#15283f;--muted:#53687e;--line:#dbe4ed;--accent:#126a76}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}
header{background:#142a42;color:white;padding:22px max(24px,calc((100vw - 1180px)/2));position:relative}
header a{color:#b8edf0}header strong{font-size:23px}header .stamp{color:#cbd8e5;font-size:13px}
nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px}main{max-width:1230px;margin:22px auto;padding:0 24px 48px}
h1{font-size:27px;margin:20px 0 8px}h2{font-size:20px;margin:28px 0 10px}h3{font-size:17px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}p{color:var(--muted)}
.table{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px;margin:12px 0 24px}
table{width:100%;border-collapse:collapse;font-size:14px}th{background:#e9f0f6;text-align:left;white-space:nowrap}td,th{padding:11px 14px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border-bottom:0}
img.plot{display:block;background:white;border:1px solid var(--line);border-radius:10px;width:100%;height:auto;margin:12px 0 24px}
.notice{padding:14px 18px;background:#e8f4f2;border-left:4px solid #168474;border-radius:4px}.small{font-size:12px;color:var(--muted)}code{background:#e9eef3;padding:2px 5px;border-radius:3px}
@media(max-width:650px){main{padding:0 12px}header{padding:18px 16px}td,th{padding:8px}h1{font-size:22px}}
"""
SCRIPT = """
let refreshing=false;
setInterval(async()=>{if(refreshing||document.hidden)return;refreshing=true;try{
const r=await fetch(location.href,{cache:'no-store'});if(!r.ok)throw Error(r.status);
const doc=new DOMParser().parseFromString(await r.text(),'text/html');
if(doc.body.dataset.revision!==document.body.dataset.revision){
const y=scrollY;document.querySelector('main').innerHTML=doc.querySelector('main').innerHTML;
document.querySelector('.stamp').innerHTML=doc.querySelector('.stamp').innerHTML;
document.body.dataset.revision=doc.body.dataset.revision;window.scrollTo(0,y);}
const health=await (await fetch('/live/service_status.json',{cache:'no-store'})).json();
document.querySelector('#connection').textContent=health.status==='ok'?'自动刷新已连接':'报告更新异常：'+(health.error||'请检查报告服务');
}catch(e){document.querySelector('#connection').textContent='更新连接中断，请检查报告服务';}finally{refreshing=false;}},10000);
"""


class Page:
    def __init__(self, title, folder, root, snapshot):
        self.title, self.folder, self.root, self.snapshot = title, folder, root, snapshot
        self.blocks = []

    def href(self, path):
        return quote(os.path.relpath(path, self.folder).replace('\\','/'), safe='/.:#')

    def paragraph(self, text):
        self.blocks.append(('p', text))

    def heading(self, text):
        self.blocks.append(('h', text))

    def link(self, label, path):
        self.blocks.append(('a', label, self.href(path)))

    def table(self, headers, rows):
        self.blocks.append(('table', headers, rows))

    def image(self, label, path):
        if path.exists():
            self.blocks.append(('img', label, self.href(path), path.stat().st_mtime_ns))

    def save(self):
        escape = html.escape
        md = [f'# {self.title}', '', f"更新时间：{self.snapshot['updated']}（北京时间）。", '']
        body = [f'<h1>{escape(self.title)}</h1>']
        for block in self.blocks:
            kind, *args = block
            if kind in ('p','h'):
                tag = 'p' if kind == 'p' else 'h2'
                body.append(f'<{tag}>{escape(args[0])}</{tag}>')
                md.extend([('## ' if kind == 'h' else '')+args[0], ''])
            elif kind == 'a':
                label, target = args
                body.append(f'<p><a href="{escape(target, quote=True)}">{escape(label)}</a></p>')
                md.extend([f'[{label}]({target.replace("index.html", "report.md")})', ''])
            elif kind == 'img':
                label, target, version = args
                body.append(f'<img class="plot" src="{escape(target)}?v={version}" alt="{escape(label)}" loading="lazy">')
                md.extend([f'![{label}]({target})', ''])
            elif kind == 'table':
                headers, rows = args
                body.append('<div class="table"><table><thead><tr>'+''.join(f'<th>{escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>')
                body.extend('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
                body.append('</tbody></table></div>')
                clean = lambda x: str(x).replace('|', '\\|').replace('\n',' ')
                md.extend(['| '+' | '.join(map(clean,headers))+' |', '| '+' | '.join('---' for _ in headers)+' |'])
                md.extend('| '+' | '.join(map(clean,row))+' |' for row in rows)
                md.append('')
        navigation = [('六任务总报告', self.root/'index.html')]+[(t,self.root/'tasks'/t/'index.html') for t in TASKS]
        nav = ''.join(f'<a href="{escape(self.href(p))}">{escape(label)}</a>' for label,p in navigation)
        page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">'
            f'<title>{escape(self.title)}</title><style>{STYLE}</style></head><body data-revision="{escape(self.snapshot["updated"])}">'
            f'<header><strong>Open-SCORE · v5.1 实验报告</strong><div class="stamp">数据刷新：{escape(self.snapshot["updated"])} 北京时间 · <span id="connection">每 10 秒自动刷新</span></div><nav>{nav}</nav></header>'
            '<main>'+''.join(body)+f'</main><script>{SCRIPT}</script></body></html>')
        atomic(self.folder/'index.html', page)
        atomic(self.folder/'report.md', '\n'.join(md))


def result_rows(methods):
    return [[m['name'], m['checkpoint'], fraction(m['test']), f"{m['test']['n']}/{m['test']['expected']}",
        '完整' if m['test']['complete'] else '未完整', f"{100*m['paired']['difference']:+.1f}" if m['paired']['n'] else '待配对',
        (f"[{100*m['paired']['interval'][0]:+.1f}, {100*m['paired']['interval'][1]:+.1f}]") if m['paired']['n'] else '待配对', m['effect']] for m in methods]


RESULT_HEADERS = ['实验臂', '固定末期检查点', '测试成功/局数', '测试覆盖', '测试状态', '比规则胜率差（百分点）', '保守95%配对区间（百分点）', '证据判断']


def comparison_plot(snap,root,cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [m for _,arms,_ in TASKS.values() for m in arms]+CONTROLS
    scores = [snap['methods'][m]['test'] for m in ids]
    key = hashlib.sha256(json.dumps([(s['wins'],s['n']) for s in scores]).encode()).hexdigest()
    if cache.get('comparison') == key and (root/'comparison.png').exists():
        return
    fig,ax = plt.subplots(figsize=(11,7))
    for i,(m,s) in enumerate(zip(ids,scores)):
        if s['n']:
            ax.barh(i,s['rate'],color='#126a76' if m not in CONTROLS else '#8998a8',alpha=1 if s['complete'] else .45)
            lo,hi=s['interval']
            ax.errorbar(s['rate'],i,xerr=[[max(0.,s['rate']-lo)],[max(0.,hi-s['rate'])]],fmt='none',color='#17283c',capsize=3)
            ax.text(min(.96,hi+.025),i,f"{s['wins']}/{s['n']}"+(' partial' if not s['complete'] else ''),va='center',fontsize=9)
        else:
            ax.text(.02,i,'Pending: no final-policy test data',va='center',fontsize=9,color='#687887')
    ax.set(yticks=range(len(ids)),yticklabels=ids,xlim=(0,1),xlabel='Native success rate (Wilson 95% interval)',title='Fixed final-policy test: 9 research arms + 5 shared controls')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    import io
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=130);plt.close(fig)
    atomic(root/'comparison.png',buffer.getvalue());cache['comparison']=key


def diagnostic_table(page, snap, task, method=None):
    rows = [r for r in snap['diagnostics'] if r.get('task_id') == task and (not method or r.get('method_id') == method)]
    page.paragraph(f"独立诊断明细来自原运行器最近一次汇总，更新时间：{snap['diagnostics_updated'] or '尚无记录'}；其刷新周期约 5 分钟。")
    if not rows:
        page.paragraph('尚无已汇总的独立诊断记录。')
        return
    groups = defaultdict(list)
    for r in rows:
        groups[r.get('method_id'), r.get('state_set') or r.get('split') or '未标记状态集'].append(r)
    out = []
    for (m, state_set), rs in groups.items():
        def avg(key):
            vals = []
            for row in rs:
                try:
                    v = float(row.get(key, ''))
                    if math.isfinite(v): vals.append(v)
                except (ValueError, TypeError): pass
            return f'{sum(vals)/len(vals):.4f}' if vals else '未记录'
        out.append([m,state_set,len(rs),avg('selected_vs_rule_verification_gain'),avg('teacher_minus_method_verification_gap'),avg('ranking_accuracy')])
    page.table(['方法','状态集','已记录状态数','选择方案－规则（复核均值）','教师－方法（复核均值）','状态内排序准确率均值'],out)
    page.paragraph('此处为已记录状态的描述性均值，不合并不同状态集，也不将模拟分支数当作独立测试回合数。')


def render(run, snap, plot_cache):
    root = run/'live'
    comparison_plot(snap,root,plot_cache)
    for m in snap['methods'].values():
        folder = root/'methods'/m['id']
        series = build_plots(m, folder, plot_cache)
        p = Page(m['name']+' · 独立实验报告', folder, root, snap)
        p.paragraph(f"所属任务 {m['task']}；{progress_text(snap['tasks'][m['task']])}。")
        if not m['training'] and not m['validation'] and not m['test']['n']:
            p.paragraph('该实验臂尚无已提交的训练或评估结果，不使用同任务中其他实验臂的结果替代。')
        if m['id'] == 't6_adv_gated':
            p.paragraph('此臂共用 ADV 价值网络，不单独训练网络；阈值由验证集选择，部署测试独立记录。')
            p.paragraph('所选阈值的验证成绩参与过阈值选择，不能代替独立测试成绩。')
            p.link('共享 ADV 网络的训练与留出指标',root/'methods'/'t6_adv'/'index.html')
        p.paragraph('训练窗口胜率、固定验证集、末期测试分别报告。缺失验证点不插值，未完成实验不记为零胜率；主表不使用验证最优模型替代固定末期模型。')
        p.heading('验证与训练过程')
        p.paragraph(validation_last(m))
        if m['validation']:
            p.table(['检查点','成功/局数','较易','中等','等人数','验证完整'],[
                [v['checkpoint'], fraction(v), *[pct(d['rate']) for d in v['difficulty']], '是' if v['complete'] else '否'] for v in m['validation']])
        if m['offline']:
            p.table(['epoch','Brier','ECE','配对MSE','排序准确率','候选选择损失'],[
                [v['epoch'], *[f'{v[k]:.4f}' if finite(v.get(k)) else '不适用/未记录' for k in
                ('brier','ece','advantage_mse','ranking_accuracy','empirical_candidate_regret')]] for v in m['offline']])
            p.paragraph('以上是留出状态的预测/排序指标，不是环境胜率。ADV 没有绝对成功概率，因此不填入 Brier/ECE。')
        if series:
            p.image('验证、训练和测试覆盖曲线', folder/'curves.png')
        else:
            p.paragraph('尚无该实验臂的曲线数据。规划与冻结规则没有梯度训练曲线，产生测试记录后展示累计测试成功率。')
        p.heading('固定末期模型的独立测试')
        p.table(RESULT_HEADERS, result_rows([m]))
        p.table(['红方','蓝方','成功/局数','胜率','Wilson95%区间','该单元完整'],[
            [g['red'],g['blue'],f"{g['wins']}/{g['n']}",pct(g['rate']),
             f"[{pct(g['interval'][0])}, {pct(g['interval'][1])}]" if g['n'] else '尚无数据',g['complete']] for g in m['test']['cells']])
        p.heading('独立候选复核')
        diagnostic_table(p,snap,m['task'],m['id'])
        p.heading('原始证据')
        if m['source']: p.link('逐局末期测试记录', run/m['source'])
        for v in m['validation']: p.link('验证 '+v['checkpoint'], run/v['source'])
        p.link('所属任务报告',root/'tasks'/m['task']/'index.html')
        p.link('复现范围及实际参数',run/m['task']/'reproduction_map.md')
        p.save()
    for task, info in snap['tasks'].items():
        p = Page(f'{task}：{info["title"]} · 实验报告',root/'tasks'/task,root,snap)
        p.paragraph(info['question'])
        p.paragraph(progress_text(info))
        p.paragraph(f"任务进度记录时间：{local_time(info['updated']) if info['updated'] else '尚无记录'}。已完成任务的进度时间保持完成时刻，不伪装成继续训练。")
        p.heading('实验设置与固定配额')
        p.paragraph(f"训练种子 {snap['config']['seed']}；{len(snap['config']['cells'])} 单元固定等权混合；已知 reactive 对手；共享规则底层；50 个物理步；无奖励塑形。方法源码 {snap['method_commit']}。")
        p.table(['参数','冻结值'],[[k,json.dumps(v,ensure_ascii=False)] for k,v in snap['config'].get(task.lower(),{}).items()])
        arms = [snap['methods'][m] for m in info['arms']]
        p.heading('各实验臂结果')
        p.table(RESULT_HEADERS,result_rows(arms))
        for m in arms:
            p.heading(m['name'])
            p.paragraph(validation_last(m))
            p.link('打开该实验臂的独立报告与原始记录',root/'methods'/m['id']/'index.html')
            p.image(m['name']+' 曲线',root/'methods'/m['id']/'curves.png')
        if task == 'T4':
            p.heading('教师与学生：同一批开局的配对验证')
            p.table(['轮次','配对局数','教师成功','学生成功','教师－学生'],[
                [r['round'],r['n'],r['teacher'],r['student'],pct((r['teacher']-r['student'])/r['n'])] for r in snap['teacher_pairs']])
            p.paragraph('只比较 family_id 相同的开局；30 局教师子集不与学生全部 150 局直接相减。')
        if task == 'T5':
            p.heading('固定归属下的纯分组诊断')
            partition = snap['partition'].get('rows',[])
            p.table(['分组方法','已复核状态数','终局成功概率的状态均值'],[
                [name,len(partition),f'{sum(float(r[key]) for r in partition)/len(partition):.4f}' if partition else '尚无数据']
                for key,name in [('learned','学习分组'),('rule','规则分组'),('grand','每目标大组'),('singletons','单体组')]])
            p.paragraph('这是固定身份和目标归属后的反事实状态诊断，不是新一批在线回合胜率。')
        if task == 'T6':
            p.heading('门控阈值：仅在验证集选择')
            p.table(['阈值','验证成功率'],[[x['threshold'],pct(x.get('success_rate'))] for x in snap['gate'].get('thresholds',[])])
            p.paragraph('BCE、ADV、ADV-gated 的末期测试分别保留，不把三者均值作为 T6 的成绩。')
        p.heading('机制诊断与证据')
        diagnostic_table(p,snap,task)
        if task in ('T1','T2'):
            p.image('独立预算诊断曲线',run/task/'budget_curves.png')
        p.link('原运行器的任务分析',run/task/'analysis.md') if (run/task/'analysis.md').exists() else None
        p.link('复现范围与参数',run/task/'reproduction_map.md')
        p.link('六任务总报告',root/'index.html')
        p.save()
    p = Page('六任务最终汇总实验报告' if snap['complete'] else '六任务实时汇总实验报告',root,root,snap)
    if snap['config'].get('smoke'):
        p.paragraph('工程短测目录：这里只验证执行流程，不用于方法有效性结论。')
    p.paragraph('全部固定实验已完成。' if snap['complete'] else '实验仍在进行：以下为当前已提交记录，尚未完成的实验臂继续显示等待或部分结果。')
    p.paragraph('训练进度、验证曲线和胜率每 10 秒读取新记录；独立候选诊断沿用原运行器约 5 分钟一次的汇总。浏览器自动更新；Markdown 和 PNG 同步写盘。全部数据仅一个训练种子，不能据此宣称跨训练种子稳健。')
    p.heading('六个独立实验报告')
    p.table(['任务','研究路线','当前阶段','已完整测试的主要臂','最新验证或留出指标'],[
        [t,info['title'],progress_text(info),f"{sum(snap['methods'][m]['test']['complete'] for m in info['arms'])}/{len(info['arms'])}",
         '；'.join(NAMES[m]+' '+validation_last(snap['methods'][m]) for m in info['arms'])] for t,info in snap['tasks'].items()])
    for t,info in snap['tasks'].items():
        p.link(t+' '+info['title']+'：独立实验报告',root/'tasks'/t/'index.html')
    p.heading('九个主要实验臂的统一末期测试')
    p.image('固定末期策略的测试成功率与95%区间',root/'comparison.png')
    p.paragraph('图中缺失实验标为 Pending，部分测试标为 partial；误差线为各臂单独的 Wilson 区间，方法差异的配对区间见下表。')
    main = [snap['methods'][m] for _,arms,_ in TASKS.values() for m in arms]
    p.table(RESULT_HEADERS,result_rows(main))
    p.heading('五个公共对照')
    p.table(RESULT_HEADERS,result_rows([snap['methods'][m] for m in CONTROLS]))
    for m in CONTROLS:
        p.link(NAMES[m]+'：独立对照报告',root/'methods'/m/'index.html')
    p.paragraph('配对区间为两个 97.5% Clopper–Pearson 区间的 Bonferroni 组合；只匹配同一 family_id。总胜率含容易任务，不能直接与 v4 等人数协议比较。')
    p.heading('分难度末期胜率')
    p.table(['方法','较易','中等','等人数'],[[m['name'],*[fraction(d) for d in m['test']['difficulty']]] for m in snap['methods'].values()])
    p.heading('各路线验证与训练曲线')
    for m in main:
        p.heading(m['name'])
        p.paragraph(validation_last(m))
        p.image(m['name']+' 曲线',root/'methods'/m['id']/'curves.png')
        p.link('独立实验臂报告',root/'methods'/m['id']/'index.html')
    p.heading('原始汇总与实验协议')
    p.link('原运行器末期测试与独立诊断汇总',run/'final_report.md')
    p.link('冻结预算清单',run/'shared/budget_manifest.json')
    if snap['warnings']:
        p.paragraph('数据读取提示：'+'；'.join(snap['warnings']))
    p.save()
    # A lightweight machine-readable companion omits raw per-episode data.
    public = dict(updated=snap['updated'], complete=snap['complete'], tasks=snap['tasks'],
        methods={k:{n:v for n,v in m.items() if n not in ('test_rows','training','offline')} for k,m in snap['methods'].items()})
    atomic(root/'snapshot.json', json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False))


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','public, max-age=86400, immutable' if self.path.split('?')[0].endswith('.png') else 'no-store')
        super().end_headers()

    def log_message(self, format, *args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','304'):
            super().log_message(format,*args)


def active_service(lock):
    import psutil
    previous = read_json(lock)
    try:
        proc = psutil.Process(previous['pid'])
        return previous if abs(proc.create_time()-previous['created']) < .01 else None
    except (psutil.NoSuchProcess,KeyError):
        return None


def acquire_service(lock, token):
    """Exclusive creation; allow a concurrent creator to finish writing its PID."""
    for _ in range(30):
        try:
            fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                json.dump(token,stream)
            return None
        except FileExistsError:
            previous = active_service(lock)
            if previous:
                return previous
            try:
                stat = lock.stat()
                if time.time()-stat.st_mtime > 3 and lock.stat().st_mtime_ns == stat.st_mtime_ns:
                    lock.unlink()
            except FileNotFoundError:
                pass
            time.sleep(.15)
    raise RuntimeError('The live report lock is being initialized; retry shortly.')


def serve(args):
    import psutil
    run = Path(args.run_dir).resolve()
    output = run/'live'
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'service.lock'
    if args.once and active_service(lock):
        raise RuntimeError('A live report service already owns this output; read its reports or stop that service before --once.')
    token = dict(pid=os.getpid(),created=psutil.Process().create_time(),url=f'http://127.0.0.1:{args.port}/live/')
    server = None
    reader, cache = Inputs(), {}
    stop = threading.Event()
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    if not args.once:
        previous = acquire_service(lock,token)
        if previous:
            print(f"Live report already running: {previous.get('url')}",flush=True)
            return 0
        from functools import partial
        try:
            server = ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(run)))
        except BaseException:
            lock.unlink(missing_ok=True)
            raise
        threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        while not stop.is_set():
            started = time.monotonic()
            try:
                snap = collect(run,reader)
                render(run,snap,cache)
                elapsed = time.monotonic()-started
                status = dict(**token,updated=local_time(),status='ok',interval_s=args.interval,
                    refresh_wall_s=round(elapsed,3),experiment_complete=snap['complete'],
                    completed_tasks=sum(t['state']=='已完成' for t in snap['tasks'].values()),
                    completed_test_arms=sum(m['test']['complete'] for m in snap['methods'].values()))
                atomic(output/'service_status.json',json.dumps(status,ensure_ascii=False,indent=2))
                print(f"[{status['updated']}] reports {status['completed_tasks']}/6 tasks; {status['completed_test_arms']}/14 final tests; refresh {elapsed:.2f}s; {token['url']}",flush=True)
            except Exception as error:
                import traceback
                atomic(output/'service_status.json',json.dumps(dict(**token,updated=local_time(),status='error',error=str(error)),ensure_ascii=False))
                traceback.print_exc()
                if args.once:
                    raise
            if args.once:
                return 0
            stop.wait(max(.5,args.interval-(time.monotonic()-started)))
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if not args.once and read_json(lock).get('pid') == os.getpid():
            lock.unlink(missing_ok=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default=str(Path(__file__).resolve().parents[1]/'outputs/v5_parallel/v5_20260907_main'))
    parser.add_argument('--interval',type=float,default=10.)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least 1 second')
    return serve(args)


if __name__ == '__main__':
    raise SystemExit(main())
