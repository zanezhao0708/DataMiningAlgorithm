/* ===== ML算法可视化实验室 · 前端主逻辑 ===== */

// 全局状态
const state = {
    task: 'classification',
    dataset: 'moons',
    nClasses: 3,
    nSamples: 300,
    noise: 0.15,
    algorithm: null,
    catalog: null,      // 算法目录 {task: [{id, name, params, ...}]}
    data: null,         // {X, y}
    result: null,       // 最近一次运行结果
    // 聚类动画
    anim: { frames: [], frame: 0, playing: false, timer: null, t0: 0 },
    abort: null,
};

// 类别配色（最多8类 + 噪声灰）
const COLORS = ['#f85149', '#3fb950', '#58a6ff', '#d29922', '#bc8cff', '#f778ba', '#39c5cf', '#ffa657'];
const NOISE_COLOR = '#6e7681';
const $ = (id) => document.getElementById(id);

// ===================== 初始化 =====================
async function init() {
    // 拉取算法目录
    const res = await fetch('/api/algorithms');
    state.catalog = await res.json();
    const total = Object.values(state.catalog).reduce((s, l) => s + l.length, 0);
    $('algo-count').textContent = `${total} 个算法`;

    bindUI();
    renderAlgoList();
    await regenerateData(true);
    // 默认选中第一个算法并运行
    const first = state.catalog[state.task][0];
    selectAlgorithm(first.id);
}

function bindUI() {
    // 任务tab
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.task = tab.dataset.task;
            renderDatasetOptions();
            renderAlgoList();
            stopAnim();
            hideTree();
            $('anim-bar').hidden = true;
            selectAlgorithm(state.catalog[state.task][0].id);
            regenerateData(true);
        });
    });

    // 数据集控件
    $('dataset-select').addEventListener('change', e => {
        state.dataset = e.target.value;
        regenerateData(true);
    });
    $('param-nsamples').addEventListener('input', e => {
        $('val-nsamples').textContent = e.target.value;
        state.nSamples = +e.target.value;
        debounceRun();
    });
    $('param-noise').addEventListener('input', e => {
        $('val-noise').textContent = (+e.target.value).toFixed(2);
        state.noise = +e.target.value;
        debounceRun();
    });
    $('param-nclasses').addEventListener('input', e => {
        $('val-nclasses').textContent = e.target.value;
        state.nClasses = +e.target.value;
        debounceRun();
    });
    $('btn-regen').addEventListener('click', () => regenerateData(true));
    $('btn-run').addEventListener('click', () => runAlgorithm());

    // 动画控制
    $('anim-play').addEventListener('click', toggleAnim);
    $('anim-prev').addEventListener('click', () => stepAnim(-1));
    $('anim-next').addEventListener('click', () => stepAnim(1));
    $('anim-slider').addEventListener('input', e => {
        stopAnim();
        state.anim.frame = +e.target.value;
        draw();
    });
}

// ===================== 渲染侧栏 =====================
function renderDatasetOptions() {
    const sel = $('dataset-select');
    sel.innerHTML = '';
    const options = {
        classification: [['moons', '双月牙'], ['circles', '同心圆'], ['xor', '异或象限'],
                         ['spiral', '双螺旋'], ['blobs2', '双高斯团']],
        clustering: [['blobs', '高斯团块'], ['moons', '双月牙'], ['circles', '同心圆']],
        regression: [['sin', '正弦波形'], ['linear', '线性']],
        dim_reduction: [['highdim', '高维高斯(10维)']],
    }[state.task];

    for (const [id, name] of options) {
        const opt = document.createElement('option');
        opt.value = id; opt.textContent = name;
        sel.appendChild(opt);
    }
    state.dataset = options[0][0];
    // 降维任务显示类别数控制，其他按需
    $('control-nclasses').hidden = !(state.task === 'dim_reduction' || state.task === 'clustering' && state.dataset === 'blobs');
}

function renderAlgoList() {
    const list = $('algo-list');
    list.innerHTML = '';
    for (const algo of state.catalog[state.task]) {
        const div = document.createElement('div');
        div.className = 'algo-item';
        div.dataset.id = algo.id;
        const tags = [];
        if (algo.has_tree) tags.push('<span class="mini-tag tree">树</span>');
        if (algo.animated) tags.push('<span class="mini-tag anim">动画</span>');
        div.innerHTML = `<span>${algo.name}</span><span class="algo-tags">${tags.join('')}</span>`;
        div.addEventListener('click', () => selectAlgorithm(algo.id));
        list.appendChild(div);
    }
}

function selectAlgorithm(id) {
    state.algorithm = id;
    document.querySelectorAll('.algo-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === id);
    });
    renderParams();
    runAlgorithm();
}

// 渲染算法参数滑块
function renderParams() {
    const algo = state.catalog[state.task].find(a => a.id === state.algorithm);
    const box = $('param-container');
    box.innerHTML = '';
    if (!algo) return;

    $('stage-title').textContent = algo.name;

    for (const p of algo.params) {
        if (p.type === 'select') {
            const c = document.createElement('div');
            c.className = 'control';
            c.innerHTML = `<label>${p.label}</label>`;
            const sel = document.createElement('select');
            for (const o of p.options) {
                const opt = document.createElement('option');
                opt.value = o; opt.textContent = o;
                if (o === p.default) opt.selected = true;
                sel.appendChild(opt);
            }
            sel.addEventListener('change', () => runAlgorithm());
            c.appendChild(sel);
            box.appendChild(c);
        } else {
            const c = document.createElement('div');
            c.className = 'control';
            c.innerHTML = `<label>${p.label} <span class="val" id="pv-${p.key}">${p.default}</span></label>
                           <input type="range" min="${p.min}" max="${p.max}"
                                  step="${p.step}" value="${p.default}">`;
            const input = c.querySelector('input');
            const fmt = (v) => p.step < 1 ? (+v).toFixed(String(p.step).split('.')[1].length) : v;
            input.addEventListener('input', () => {
                $('pv-' + p.key).textContent = fmt(input.value);
            });
            input.addEventListener('change', () => runAlgorithm());
            box.appendChild(c);
        }
    }
}

// 收集当前算法参数
function collectParams() {
    const algo = state.catalog[state.task].find(a => a.id === state.algorithm);
    const params = {};
    for (const p of (algo ? algo.params : [])) {
        params[p.key] = p.type === 'select' ? p.default : p.default;
    }
    // 从DOM读取实际值（renderParams渲染的默认值）
    document.querySelectorAll('#param-container input[type=range]').forEach(input => {
        params[input.closest('.control').querySelector('.val').id.replace('pv-', '')] = +input.value;
    });
    return params;
}

// ===================== 数据与运行 =====================
async function regenerateData(withRun) {
    const body = {
        task: state.task, dataset: state.dataset,
        n_samples: state.nSamples, noise: state.noise,
        n_classes: state.nClasses, seed: Math.floor(Math.random() * 1e6),
    };
    const res = await fetch('/api/dataset', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    state.data = await res.json();
    if (withRun && state.algorithm) runAlgorithm();
    else draw();
}

let debounceTimer = null;
function debounceRun() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        await regenerateData(false);
        if (state.algorithm) runAlgorithm();
    }, 350);
}

async function runAlgorithm() {
    if (!state.algorithm || !state.data) return;
    if (state.abort) state.abort.abort();
    state.abort = new AbortController();

    $('btn-run').disabled = true;
    $('loading').hidden = false;
    const t0 = performance.now();

    try {
        const body = {
            task: state.task, algorithm: state.algorithm,
            params: collectParams(),
            X: state.data.X,
            y: state.task === 'clustering' ? state.data.X.map(() => 0) : state.data.y,
        };
        // 聚类无需标签，传null
        if (state.task === 'clustering') body.y = null;

        const res = await fetch('/api/run', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body), signal: state.abort.signal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        state.result = await res.json();
        state.elapsed = performance.now() - t0;

        stopAnim();
        const algo = state.catalog[state.task].find(a => a.id === state.algorithm);

        // 聚类动画
        if (state.task === 'clustering' && state.result.frames) {
            state.anim.frames = state.result.frames;
            state.anim.frame = 0;
            $('anim-bar').hidden = false;
            $('anim-slider').max = state.result.frames.length - 1;
            $('anim-slider').value = 0;
            playAnim();
        } else {
            $('anim-bar').hidden = true;
            draw();
        }

        // 决策树
        if (state.result.tree) renderTree(state.result.tree);
        else hideTree();

        renderMetrics();
    } catch (e) {
        if (e.name !== 'AbortError') {
            $('metrics').innerHTML = `<div class="metric"><div class="m-label">错误</div>
                <div class="m-value" style="color:var(--danger);font-size:12px">${e.message.slice(0, 40)}</div></div>`;
        }
    } finally {
        $('btn-run').disabled = false;
        $('loading').hidden = true;
    }
}

function renderMetrics() {
    const r = state.result;
    const box = $('metrics');
    const cards = [];
    const add = (label, value) => cards.push(
        `<div class="metric"><div class="m-label">${label}</div><div class="m-value">${value}</div></div>`);

    if (state.task === 'classification') add('准确率', (r.accuracy * 100).toFixed(1) + '%');
    if (state.task === 'clustering') {
        add('轮廓系数', r.silhouette.toFixed(3));
        add('发现簇数', r.n_found);
    }
    if (state.task === 'regression') add('R²', r.r2.toFixed(3));
    if (state.task === 'dim_reduction') {
        if (r.explained_total != null) add('方差解释率', (r.explained_total * 100).toFixed(1) + '%');
        if (r.kl != null) add('KL散度', r.kl.toFixed(3));
    }
    add('耗时', (state.elapsed / 1000).toFixed(2) + 's');
    box.innerHTML = cards.join('');
}

// ===================== 画布绘制 =====================
const canvas = $('main-canvas');
const ctx = canvas.getContext('2d');

function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: rect.width, h: rect.height };
}

// 等比坐标变换
function makeTransform(X, w, h, padding = 20) {
    const xs = X.map(p => p[0]), ys = X.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
    const scale = Math.min((w - padding * 2) / dx, (h - padding * 2) / dy);
    const ox = (w - dx * scale) / 2, oy = (h - dy * scale) / 2;
    return {
        sx: x => ox + (x - x0) * scale,
        sy: y => h - (oy + (y - y0) * scale),  // y轴翻转
        x0, x1, y0, y1, scale,
        inv: (px, py) => [x0 + (px - ox) / scale, y0 + (h - py - oy) / scale],
    };
}

function drawGrid(t) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.strokeStyle = 'rgba(48, 54, 61, .5)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
        const gx = (w / 8) * i, gy = (h / 8) * i;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
}

function drawPoints(X, y, t, r = 5) {
    for (let i = 0; i < X.length; i++) {
        const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[(y[i] ?? 0) % COLORS.length];
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#0d1117';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function draw() {
    const { w, h } = setupCanvas();
    ctx.clearRect(0, 0, w, h);
    drawGrid();
    if (!state.data) return;

    const X = state.data.X;

    if (state.task === 'classification') drawClassification(X, w, h);
    else if (state.task === 'clustering') drawClusteringFrame(X, w, h);
    else if (state.task === 'regression') drawRegression(X, w, h);
    else if (state.task === 'dim_reduction') drawEmbedding(X, w, h);
}

// ---- 分类：决策边界 ----
function drawClassification(X, w, h) {
    const t = makeTransform(X, w, h);
    const r = state.result;

    if (r && r.grid) {
        // 决策边界网格填色
        const g = r.grid_size;
        const cw = w / g, ch = h / g;
        // 扩展绘制范围（边界超出数据范围0.5）
        const [gx0, gx1] = r.x_range, [gy0, gy1] = r.y_range;
        for (let row = 0; row < g; row++) {
            for (let col = 0; col < g; col++) {
                const label = r.grid[row * g + col];
                const wx = gx0 + (gx1 - gx0) * (col + 0.5) / g;
                const wy = gy0 + (gy1 - gy0) * (row + 0.5) / g;
                const px = t.sx(wx), py = t.sy(wy);
                ctx.fillStyle = COLORS[label % COLORS.length];
                ctx.globalAlpha = 0.13;
                ctx.fillRect(px - cw, py - ch, cw * 2 + 1, ch * 2 + 1);
            }
        }
        ctx.globalAlpha = 1;
    }
    drawPoints(X, state.data.y, t);
}

// ---- 聚类：支持动画帧 ----
function drawClusteringFrame(X, w, h) {
    const anim = state.anim;
    const r = state.result;

    if (r && r.frames && anim.frames.length) {
        // 当前帧质心 + 按最近质心分配颜色
        const frame = anim.frames[anim.frame];
        const centroids = frame.centroids;
        const t = makeTransform(X, w, h);

        // 数据点按最近质心着色
        for (let i = 0; i < X.length; i++) {
            let best = 0, bd = Infinity;
            for (let c = 0; c < centroids.length; c++) {
                const d = (X[i][0] - centroids[c][0]) ** 2 + (X[i][1] - centroids[c][1]) ** 2;
                if (d < bd) { bd = d; best = c; }
            }
            const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fillStyle = COLORS[best % COLORS.length];
            ctx.globalAlpha = 0.9;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.strokeStyle = '#0d1117';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // 画质心轨迹（历史帧的连线）
        ctx.strokeStyle = 'rgba(230, 237, 243, .3)';
        ctx.setLineDash([4, 4]);
        for (let c = 0; c < centroids.length; c++) {
            ctx.beginPath();
            for (let f = 0; f <= anim.frame; f++) {
                const ct = anim.frames[f].centroids[c];
                const px = t.sx(ct[0]), py = t.sy(ct[1]);
                f === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            }
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // 当前质心
        for (let c = 0; c < centroids.length; c++) {
            const px = t.sx(centroids[c][0]), py = t.sy(centroids[c][1]);
            ctx.beginPath();
            ctx.arc(px, py, 10, 0, Math.PI * 2);
            ctx.fillStyle = '#0d1117';
            ctx.fill();
            ctx.strokeStyle = COLORS[c % COLORS.length];
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        $('anim-label').textContent = `迭代 ${anim.frame}`;
        $('anim-slider').value = anim.frame;
        return;
    }

    // 静态聚类结果
    const t = makeTransform(X, w, h);
    const labels = r ? r.labels : state.data.X.map(() => 0);
    const noise = r ? (r.noise || []) : [];

    for (let i = 0; i < X.length; i++) {
        const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = noise[i] ? NOISE_COLOR : COLORS[labels[i] % COLORS.length];
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#0d1117';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
    if (r && r.centers) {
        for (let c = 0; c < r.centers.length; c++) {
            const px = t.sx(r.centers[c][0]), py = t.sy(r.centers[c][1]);
            ctx.beginPath();
            ctx.arc(px, py, 10, 0, Math.PI * 2);
            ctx.fillStyle = '#0d1117';
            ctx.fill();
            ctx.strokeStyle = COLORS[c % COLORS.length];
            ctx.lineWidth = 3;
            ctx.stroke();
        }
    }
}

// ---- 回归：拟合曲线 ----
function drawRegression(X, w, h) {
    // 回归数据是一维特征，用(特征x, 目标y)构造二维点计算坐标变换
    const pts = X.map((p, i) => [p[0], state.data.y[i]]);
    const t = makeTransform(pts, w, h);
    const r = state.result;

    // 数据点
    for (let i = 0; i < pts.length; i++) {
        const px = t.sx(pts[i][0]), py = t.sy(pts[i][1]);
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#58a6ff';
        ctx.globalAlpha = 0.75;
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    // 拟合曲线
    if (r && r.curve) {
        ctx.beginPath();
        r.curve.forEach(([x, y], i) => {
            const px = t.sx(x), py = t.sy(y);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.strokeStyle = '#ffa657';
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.stroke();
    }
}

// ---- 降维：投影散点 ----
function drawEmbedding(X, w, h) {
    const r = state.result;
    if (!r || !r.embedding) return drawPoints(X, state.data.y.map(() => 0), makeTransform(X, w, h));
    const t = makeTransform(r.embedding, w, h, 30);
    drawPoints(r.embedding, state.data.y, t, 6);
}

// ===================== 动画控制 =====================
function playAnim() {
    if (state.anim.playing) return;
    state.anim.playing = true;
    $('anim-play').textContent = '⏸';
    const stepMs = 500;
    state.anim.timer = setInterval(() => {
        state.anim.frame++;
        if (state.anim.frame >= state.anim.frames.length - 1) {
            stopAnim();
            state.anim.frame = state.anim.frames.length - 1;
        }
        draw();
    }, stepMs);
}

function stopAnim() {
    state.anim.playing = false;
    $('anim-play').textContent = '▶';
    if (state.anim.timer) { clearInterval(state.anim.timer); state.anim.timer = null; }
}

function toggleAnim() {
    if (state.anim.playing) {
        stopAnim();
    } else {
        if (state.anim.frame >= state.anim.frames.length - 1) state.anim.frame = 0;
        playAnim();
    }
}

function stepAnim(delta) {
    stopAnim();
    state.anim.frame = Math.max(0, Math.min(state.anim.frames.length - 1, state.anim.frame + delta));
    draw();
}

// ===================== 决策树SVG =====================
function renderTree(root) {
    const panel = $('tree-panel');
    const container = $('tree-container');
    panel.hidden = false;

    // 布局：统计叶节点，叶子均匀分布x
    let leafX = 0;
    const NODE_W = 76, NODE_H = 34, GAP_X = 20, GAP_Y = 64;

    function layout(node, depth) {
        if (node.leaf) {
            node._x = leafX * (NODE_W + GAP_X);
            leafX++;
        } else {
            layout(node.left, depth + 1);
            layout(node.right, depth + 1);
            node._x = (node.left._x + node.right._x) / 2;
        }
        node._y = depth * (NODE_H + GAP_Y);
    }
    layout(root, 0);

    const totalW = Math.max(leafX * (NODE_W + GAP_X), 200);
    const totalH = (maxDepth(root) + 1) * (NODE_H + GAP_Y);
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', totalW);
    svg.setAttribute('height', totalH);
    svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

    const colorOf = v => COLORS[v % COLORS.length];

    function drawNode(node) {
        if (!node.leaf) {
            drawNode(node.left);
            drawNode(node.right);
            // 边
            for (const [child, label] of [[node.left, '≤'], [node.right, '>']]) {
                const path = document.createElementNS(svgNS, 'path');
                const x1 = node._x + NODE_W / 2, y1 = node._y + NODE_H;
                const x2 = child._x + NODE_W / 2, y2 = child._y;
                const my = (y1 + y2) / 2;
                path.setAttribute('d', `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`);
                path.setAttribute('class', 'tedge');
                path.setAttribute('stroke', colorOf(0) === colorOf(1) ? '#30363d' : 'rgba(88,166,255,.35)');
                svg.appendChild(path);
                const text = document.createElementNS(svgNS, 'text');
                text.setAttribute('x', (x1 + x2) / 2);
                text.setAttribute('y', my - 3);
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('class', 'tedge-label');
                text.textContent = label;
                svg.appendChild(text);
            }
        }
        const g = document.createElementNS(svgNS, 'g');
        g.setAttribute('class', 'tnode' + (node.leaf ? ' leaf' : ''));
        const rect = document.createElementNS(svgNS, 'rect');
        rect.setAttribute('x', node._x); rect.setAttribute('y', node._y);
        rect.setAttribute('width', NODE_W); rect.setAttribute('height', NODE_H);
        rect.setAttribute('rx', 6);
        if (node.leaf) rect.setAttribute('stroke', colorOf(node.value));
        g.appendChild(rect);

        const text = document.createElementNS(svgNS, 'text');
        text.setAttribute('x', node._x + NODE_W / 2);
        text.setAttribute('y', node._y + 21);
        text.setAttribute('text-anchor', 'middle');
        text.textContent = node.leaf ? `类别 ${node.value}` : `${node.feature} ${node.threshold}`;
        if (node.leaf) text.setAttribute('fill', colorOf(node.value));
        g.appendChild(text);
        svg.appendChild(g);
    }
    drawNode(root);

    container.innerHTML = '';
    container.appendChild(svg);
}

function maxDepth(node) {
    if (node.leaf) return 0;
    return 1 + Math.max(maxDepth(node.left), maxDepth(node.right));
}

function hideTree() { $('tree-panel').hidden = true; }

// ===================== 启动 =====================
window.addEventListener('resize', draw);
init();
