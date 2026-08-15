/* ===== ML算法可视化实验室 · 前端主逻辑（TF Playground 风格） ===== */

const state = {
    task: 'classification',
    dataset: 'moons',
    nClasses: 3,
    nSamples: 300,
    noise: 0.15,
    testRatio: 0.2,
    algorithm: null,
    catalog: null,
    previews: null,     // 数据集缩略图数据
    data: null,         // {X, y}
    result: null,
    elapsed: 0,
    anim: { frames: [], frame: 0, playing: false, timer: null, type: null, speed: 1 },
    abort: null,
    pendingRun: null,   // 请求合并：防止滑块拖动时请求堆积
    runToken: 0,        // 请求序号：丢弃乱序返回的过期响应
    userSelected: false, // 用户在tab初始化期间手动选择过算法
};

// 类别配色（TF Playground 橙蓝配色）
const COLORS = ['#e6850f', '#1a73e8', '#34a853', '#9c27b0', '#f06292', '#00897b', '#c62828', '#f9a825'];
const NOISE_COLOR = '#9e9e9e';
const $ = (id) => document.getElementById(id);

// ===================== 初始化 =====================
async function init() {
    const res = await fetch('/api/algorithms');
    state.catalog = await res.json();
    const total = Object.values(state.catalog).reduce((s, l) => s + l.length, 0);
    $('algo-count').textContent = `${total} 个算法`;

    bindUI();
    await loadPreviews();
    renderAlgoList();
    await regenerateData(false);
    selectAlgorithm(state.catalog[state.task][0].id);
}

function bindUI() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.task = tab.dataset.task;
            state.userSelected = false;
            stopAnim();
            hideTree();
            $('anim-bar').hidden = true;
            $('chart-panel').hidden = true;
            $('control-testratio').hidden = state.task !== 'classification';
            loadPreviews().then(() => {
                renderAlgoList();
                // 加载期间用户已手动选择算法时，不再强制重置为第一个
                if (!state.userSelected) selectAlgorithm(state.catalog[state.task][0].id);
                state.userSelected = false;
                regenerateData(true);
            });
        });
    });

    $('param-nsamples').addEventListener('input', e => {
        $('val-nsamples').textContent = e.target.value;
        state.nSamples = +e.target.value;
        scheduleRerun();
    });
    $('param-noise').addEventListener('input', e => {
        $('val-noise').textContent = (+e.target.value).toFixed(2);
        state.noise = +e.target.value;
        scheduleRerun();
    });
    $('param-nclasses').addEventListener('input', e => {
        $('val-nclasses').textContent = e.target.value;
        state.nClasses = +e.target.value;
        scheduleRerun();
    });
    $('param-testratio').addEventListener('input', e => {
        state.testRatio = +e.target.value;
        $('val-testratio').textContent = Math.round(state.testRatio * 100) + '%';
        scheduleRerun();
    });
    $('btn-reset').addEventListener('click', () => regenerateData(true));
    $('btn-play').addEventListener('click', () => {
        // 播放按钮：有动画则播放，否则重跑
        if (state.anim.frames.length) toggleAnim();
        else runAlgorithm();
    });

    $('anim-play').addEventListener('click', toggleAnim);
    $('anim-prev').addEventListener('click', () => stepAnim(-1));
    $('anim-next').addEventListener('click', () => stepAnim(1));
    $('anim-speed').addEventListener('click', cycleSpeed);
    $('anim-slider').addEventListener('input', e => {
        stopAnim();
        state.anim.frame = +e.target.value;
        draw();
    });

    $('btn-source').addEventListener('click', toggleSource);
    $('source-close').addEventListener('click', () => setSourceOpen(false));
    $('source-overlay').addEventListener('click', e => {
        // 点击遮罩空白处关闭弹窗
        if (e.target === $('source-overlay')) setSourceOpen(false);
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !$('source-overlay').hidden) setSourceOpen(false);
    });
    $('btn-issue').addEventListener('click', openIssue);
}

// ===================== 源码弹窗 =====================
const GITHUB_REPO = 'https://github.com/zanezhao0708/DataMiningAlgorithm';

function setSourceOpen(open) {
    $('source-overlay').hidden = !open;
    $('btn-source').classList.toggle('active', open);
    if (open) loadSource();
}

function toggleSource() {
    setSourceOpen($('source-overlay').hidden);
}

async function loadSource() {
    if (!state.algorithm) return;
    $('source-code').textContent = '加载中...';
    try {
        const res = await fetch(`/api/source/${state.algorithm}`);
        if (!res.ok) {
            $('source-code').textContent = `源码加载失败（HTTP ${res.status}）`;
            return;
        }
        const d = await res.json();
        $('source-path').textContent = d.path;
        $('source-code').textContent = d.source;
        $('source-github').href = `${GITHUB_REPO}/blob/main/${d.path}`;
    } catch (e) {
        $('source-code').textContent = '源码加载失败，请检查网络后重试';
    }
}

// 问题反馈：跳转 GitHub 新建 Issue，自动带上当前环境信息
function openIssue() {
    const algo = state.algorithm || '未选择';
    const body = [
        '## 问题描述', '', '',
        '## 复现步骤', '1. ', '',
        '## 期望行为', '', '',
        '---',
        `> 自动附带的环境信息：任务=${state.task}，算法=${algo}，` +
        `数据集=${state.dataset}，样本数=${state.nSamples}，浏览器=${navigator.userAgent.slice(0, 80)}`,
    ].join('\n');
    const url = `${GITHUB_REPO}/issues/new?title=${encodeURIComponent('[反馈] ')}&body=${encodeURIComponent(body)}`;
    window.open(url, '_blank');
}

// 请求合并：滑块连续拖动时只执行最后一次
function scheduleRerun() {
    clearTimeout(state.pendingRun);
    state.pendingRun = setTimeout(async () => {
        await regenerateData(false);
        if (state.algorithm) runAlgorithm();
    }, 300);
}

// ===================== 数据集缩略图 =====================
async function loadPreviews() {
    const res = await fetch(`/api/previews/${state.task}`);
    state.previews = await res.json();
    renderDatasetGrid();
}

function renderDatasetGrid() {
    const grid = $('dataset-grid');
    grid.innerHTML = '';
    const names = {
        classification: { moons: '双月牙', circles: '同心圆', xor: '异或', spiral: '双螺旋', blobs2: '双团块' },
        clustering: { blobs: '团块', moons: '月牙', circles: '同心圆' },
        regression: { sin: '正弦', linear: '线性' },
        dim_reduction: { highdim: '高维' },
    }[state.task] || {};

    for (const [id, pv] of Object.entries(state.previews)) {
        const div = document.createElement('div');
        div.className = 'ds-thumb' + (id === state.dataset ? ' active' : '');
        div.dataset.id = id;

        const cv = document.createElement('canvas');
        cv.width = 88; cv.height = 56;
        div.appendChild(cv);
        const label = document.createElement('div');
        label.className = 'ds-name';
        label.textContent = names[id] || id;
        div.appendChild(label);

        div.addEventListener('click', () => {
            state.dataset = id;
            document.querySelectorAll('.ds-thumb').forEach(el =>
                el.classList.toggle('active', el.dataset.id === id));
            regenerateData(true);
        });
        grid.appendChild(div);
        drawThumb(cv, pv.X, pv.y);
    }
    // 确保当前选中有效
    if (!state.previews[state.dataset]) {
        state.dataset = Object.keys(state.previews)[0];
        document.querySelector('.ds-thumb')?.classList.add('active');
    }
    $('control-nclasses').hidden = !(state.task === 'dim_reduction' ||
        (state.task === 'clustering' && state.dataset === 'blobs'));
}

function drawThumb(cv, X, y) {
    const ctx = cv.getContext('2d');
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);
    if (!X || !X.length) return;

    const xs = X.map(p => p[0]), ys = X.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
    const scale = Math.min((w - 8) / dx, (h - 8) / dy);
    const ox = (w - dx * scale) / 2, oy = (h - dy * scale) / 2;

    for (let i = 0; i < X.length; i++) {
        const px = ox + (X[i][0] - x0) * scale;
        const py = h - (oy + (X[i][1] - y0) * scale);
        ctx.beginPath();
        ctx.arc(px, py, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[(y[i] ?? 0) % COLORS.length];
        ctx.fill();
    }
}

// ===================== 渲染侧栏 =====================
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
    state.userSelected = true;
    document.querySelectorAll('.algo-item').forEach(el =>
        el.classList.toggle('active', el.dataset.id === id));
    renderParams();
    if (!$('source-panel').hidden) loadSource();
    runAlgorithm();
}

function renderParams() {
    const algo = state.catalog[state.task].find(a => a.id === state.algorithm);
    const box = $('param-container');
    box.innerHTML = '';
    if (!algo) return;
    $('stage-title').textContent = algo.name;

    for (const p of algo.params) {
        const c = document.createElement('div');
        c.className = 'control';
        c.dataset.key = p.key;
        if (p.type === 'select') {
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
        } else {
            const fmt = v => p.step < 1 ? (+v).toFixed(String(p.step).split('.')[1].length) : v;
            c.innerHTML = `<label>${p.label} <span class="val" id="pv-${p.key}">${fmt(p.default)}</span></label>
                           <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">`;
            const input = c.querySelector('input');
            input.addEventListener('input', () => { $('pv-' + p.key).textContent = fmt(input.value); });
            input.addEventListener('change', () => runAlgorithm());
        }
        box.appendChild(c);
    }
}

function collectParams() {
    const algo = state.catalog[state.task].find(a => a.id === state.algorithm);
    const params = {};
    for (const p of (algo ? algo.params : [])) params[p.key] = p.default;
    // 按 data-key 精确读取每个控件的当前值
    document.querySelectorAll('#param-container .control').forEach(c => {
        const key = c.dataset.key;
        if (!key) return;
        const range = c.querySelector('input[type=range]');
        const sel = c.querySelector('select');
        if (range) params[key] = +range.value;
        else if (sel) params[key] = sel.value;
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
    try {
        const res = await fetch('/api/dataset', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.data = await res.json();
    } catch (e) {
        // 数据生成失败时保留现有数据并提示，避免未处理 rejection
        $('metrics').innerHTML = `<div class="metric"><div class="m-label">错误</div>
            <div class="m-value" style="color:var(--danger);font-size:11px">数据生成失败</div></div>`;
        return;
    }
    if (withRun && state.algorithm) runAlgorithm();
    else draw();
}

async function runAlgorithm() {
    if (!state.algorithm || !state.data) return;
    if (state.abort) state.abort.abort();
    state.abort = new AbortController();
    const token = ++state.runToken;

    $('loading').hidden = false;
    const t0 = performance.now();

    try {
        const body = {
            task: state.task, algorithm: state.algorithm,
            params: collectParams(),
            X: state.data.X,
            y: state.task === 'clustering' ? null : state.data.y,
            test_ratio: state.testRatio,
        };
        const res = await fetch('/api/run', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body), signal: state.abort.signal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        // 乱序防护：仅当这是最新一次请求时才应用结果
        if (token !== state.runToken) return;
        state.result = data;
        state.elapsed = performance.now() - t0;

        stopAnim();
        // 动画帧（聚类过程 / 分类边界演化 / 回归拟合演化 / 降维投影演化）
        if (state.result.frames && state.result.frames.length > 1) {
            state.anim.frames = state.result.frames;
            state.anim.frame = 0;  // 默认停在初始未执行状态，等待用户点击播放
            state.anim.type = state.task === 'clustering' ? 'cluster'
                : state.task === 'regression' ? 'reg'
                : state.task === 'dim_reduction' ? 'embed' : 'grid';
            $('anim-bar').hidden = false;
            $('anim-slider').max = state.result.frames.length - 1;
            $('anim-slider').value = 0;
        } else {
            $('anim-bar').hidden = true;
        }
        draw();

        // 损失曲线：从初始状态开始播放时只显示起点
        if (state.result.history && state.result.history.length > 1)
            drawLossChart(state.result.frames ? state.result.history.slice(0, 1) : state.result.history);
        else $('chart-panel').hidden = true;

        if (state.result.tree) renderTree(state.result.tree);
        else hideTree();

        renderMetrics();
    } catch (e) {
        if (e.name !== 'AbortError' && token === state.runToken) {
            $('metrics').innerHTML = `<div class="metric"><div class="m-label">错误</div>
                <div class="m-value" style="color:var(--danger);font-size:11px">${e.message.slice(0, 40)}</div></div>`;
        }
    } finally {
        // 仅最新请求有权收起加载指示器（过期请求不得干扰新请求的状态）
        if (token === state.runToken) $('loading').hidden = true;
    }
}

function renderMetrics() {
    const r = state.result;
    const box = $('metrics');
    const cards = [];
    const add = (label, value) => cards.push(
        `<div class="metric"><div class="m-label">${label}</div><div class="m-value">${value}</div></div>`);

    if (state.task === 'classification') {
        add('训练准确率', (r.accuracy * 100).toFixed(1) + '%');
        if (r.test_accuracy != null) add('测试准确率', (r.test_accuracy * 100).toFixed(1) + '%');
    }
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

function makeTransform(X, w, h, padding = 20) {
    const xs = X.map(p => p[0]), ys = X.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
    const scale = Math.min((w - padding * 2) / dx, (h - padding * 2) / dy);
    const ox = (w - dx * scale) / 2, oy = (h - dy * scale) / 2;
    return {
        sx: x => ox + (x - x0) * scale,
        sy: y => h - (oy + (y - y0) * scale),
    };
}

function drawGrid() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.strokeStyle = 'rgba(0,0,0,.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
        const gx = (w / 8) * i, gy = (h / 8) * i;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
}

// 测试集点用空心标记区分
function drawPoints(X, y, t, testSet, r = 5) {
    for (let i = 0; i < X.length; i++) {
        const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
        const isTest = testSet && testSet.has(i);
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        if (isTest) {
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = COLORS[(y[i] ?? 0) % COLORS.length];
            ctx.lineWidth = 2;
            ctx.stroke();
        } else {
            ctx.fillStyle = COLORS[(y[i] ?? 0) % COLORS.length];
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }
}

function draw() {
    const { w, h } = setupCanvas();
    ctx.clearRect(0, 0, w, h);
    drawGrid();
    if (!state.data) return;

    const X = state.data.X;
    if (state.task === 'classification') drawClassification(X, w, h);
    else if (state.task === 'clustering') drawClustering(X, w, h);
    else if (state.task === 'regression') drawRegression(X, w, h);
    else if (state.task === 'dim_reduction') drawEmbedding(X, w, h);
}

// ---- 分类：决策边界（支持MLP动画帧） ----
function drawClassification(X, w, h) {
    const t = makeTransform(X, w, h);
    const r = state.result;
    const testSet = r && r.test_indices ? new Set(r.test_indices) : null;

    // 训练动画（MLP / 梯度下降）：使用当前帧的网格
    let grid = r ? r.grid : null;
    let ranges = r ? [r.x_range, r.y_range] : null;
    if (r && r.frames && state.anim.type === 'grid') {
        const frame = r.frames[state.anim.frame];
        grid = frame.grid;
        const prefix = state.algorithm === 'mlp' ? 'epoch' : '迭代';
        $('anim-label').textContent = frame.label ?? `${prefix} ${frame.epoch}`;
        $('anim-slider').value = state.anim.frame;
    }

    if (grid && ranges) {
        const g = r.grid_size;
        const cw = w / g, ch = h / g;
        const [gx0, gx1] = ranges[0], [gy0, gy1] = ranges[1];
        for (let row = 0; row < g; row++) {
            for (let col = 0; col < g; col++) {
                const label = grid[row * g + col];
                const wx = gx0 + (gx1 - gx0) * (col + 0.5) / g;
                const wy = gy0 + (gy1 - gy0) * (row + 0.5) / g;
                const px = t.sx(wx), py = t.sy(wy);
                ctx.fillStyle = COLORS[label % COLORS.length];
                ctx.globalAlpha = 0.16;
                // 半格宽的色块恰好铺满，重叠会导致半透明叠加出现色带
                ctx.fillRect(px - cw / 2, py - ch / 2, cw + 0.5, ch + 0.5);
            }
        }
        ctx.globalAlpha = 1;
    }
    drawPoints(X, state.data.y, t, testSet);
}

// ---- 聚类：支持质心轨迹 / 标签演化 / DBSCAN扩展 / GMM椭圆 动画帧 ----
function drawClustering(X, w, h) {
    const anim = state.anim;
    const r = state.result;
    const t = makeTransform(X, w, h);

    if (r && r.frames && anim.frames.length && anim.type === 'cluster') {
        const frame = anim.frames[anim.frame];
        const centroids = frame.centroids || null;

        for (let i = 0; i < X.length; i++) {
            // 优先使用帧内快照标签；否则回退到最近质心分配
            let label = 0;
            if (frame.labels != null) {
                label = frame.labels[i];
            } else if (centroids) {
                let bd = Infinity;
                for (let c = 0; c < centroids.length; c++) {
                    const d = (X[i][0] - centroids[c][0]) ** 2 + (X[i][1] - centroids[c][1]) ** 2;
                    if (d < bd) { bd = d; label = c; }
                }
            }
            const isNoise = frame.noise && frame.noise[i];
            const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fillStyle = isNoise ? NOISE_COLOR : COLORS[label % COLORS.length];
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // GMM：绘制每个成分的2σ高斯椭圆
        if (frame.gaussians) {
            for (let g = 0; g < frame.gaussians.length; g++)
                drawGaussianEllipse(frame.gaussians[g], g, t);
        }

        // 质心轨迹
        if (centroids) {
            ctx.strokeStyle = 'rgba(0,0,0,.25)';
            ctx.setLineDash([4, 4]);
            for (let c = 0; c < centroids.length; c++) {
                ctx.beginPath();
                for (let f = 0; f <= anim.frame; f++) {
                    const ct = anim.frames[f].centroids && anim.frames[f].centroids[c];
                    if (!ct) continue;
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
                ctx.arc(px, py, 9, 0, Math.PI * 2);
                ctx.fillStyle = '#fff';
                ctx.fill();
                ctx.strokeStyle = COLORS[c % COLORS.length];
                ctx.lineWidth = 3;
                ctx.stroke();
            }
        }

        $('anim-label').textContent = frame.label ?? `帧 ${anim.frame}`;
        $('anim-slider').value = anim.frame;
        return;
    }

    // 静态聚类结果
    const labels = r ? r.labels : X.map(() => 0);
    const noise = r ? (r.noise || []) : [];

    for (let i = 0; i < X.length; i++) {
        const px = t.sx(X[i][0]), py = t.sy(X[i][1]);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = noise[i] ? NOISE_COLOR : COLORS[labels[i] % COLORS.length];
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    if (r && r.centers) {
        for (let c = 0; c < r.centers.length; c++) {
            const px = t.sx(r.centers[c][0]), py = t.sy(r.centers[c][1]);
            ctx.beginPath();
            ctx.arc(px, py, 9, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = COLORS[c % COLORS.length];
            ctx.lineWidth = 3;
            ctx.stroke();
        }
    }
}

// 绘制二维高斯成分的2σ置信椭圆
function drawGaussianEllipse(g, idx, t) {
    const cov = g.cov;
    const a = cov[0][0], b = cov[0][1], c = cov[1][1];
    const tr = a + c, det = a * c - b * b;
    const disc = Math.max(0, tr * tr / 4 - det);
    const l1 = Math.max(1e-9, tr / 2 + Math.sqrt(disc));
    const l2 = Math.max(1e-9, tr / 2 - Math.sqrt(disc));
    const angle = Math.abs(b) < 1e-9 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);

    const sc = t.sx(1) - t.sx(0);  // 数据单位 → 像素（等比变换）
    const px = t.sx(g.mean[0]), py = t.sy(g.mean[1]);
    // 半径上限防溢出画布（协方差病态增大时椭圆可能失控）
    const maxR = 1.5 * Math.max(canvas.width, canvas.height);
    const rx = Math.min(2 * Math.sqrt(l1) * sc, maxR);
    const ry = Math.min(2 * Math.sqrt(l2) * sc, maxR);
    ctx.beginPath();
    ctx.ellipse(px, py, rx, ry, -angle, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS[idx % COLORS.length];
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
}

// ---- 回归：拟合曲线 ----
function drawRegression(X, w, h) {
    const pts = X.map((p, i) => [p[0], state.data.y[i]]);
    const t = makeTransform(pts, w, h);
    const r = state.result;

    // 训练动画：使用当前帧的拟合曲线
    let curve = r ? r.curve : null;
    if (r && r.frames && state.anim.type === 'reg') {
        const frame = r.frames[state.anim.frame];
        curve = frame.curve;
        $('anim-label').textContent = frame.label ?? `迭代 ${frame.epoch}`;
        $('anim-slider').value = state.anim.frame;
    }

    for (let i = 0; i < pts.length; i++) {
        const px = t.sx(pts[i][0]), py = t.sy(pts[i][1]);
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#1a73e8';
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    if (curve) {
        ctx.beginPath();
        curve.forEach(([x, y], i) => {
            const px = t.sx(x), py = t.sy(y);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.strokeStyle = '#e6850f';
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.stroke();
    }
}

// ---- 降维：投影散点（支持嵌入演化动画帧） ----
function drawEmbedding(X, w, h) {
    const r = state.result;
    if (!r || !r.embedding) return;
    let emb = r.embedding;
    if (r.frames && state.anim.frames.length && state.anim.type === 'embed') {
        const frame = r.frames[state.anim.frame];
        if (frame && frame.embedding) {
            emb = frame.embedding;
            $('anim-label').textContent = frame.label ?? `迭代 ${frame.step}`;
            $('anim-slider').value = state.anim.frame;
        }
    }
    const t = makeTransform(emb, w, h, 30);
    drawPoints(emb, state.data.y, t, null, 6);
}

// ===================== 损失曲线 =====================
function drawLossChart(history) {
    const panel = $('chart-panel');
    panel.hidden = false;
    const cv = $('loss-canvas');
    const dpr = window.devicePixelRatio || 1;
    const w = cv.parentElement.clientWidth - 24;
    cv.width = w * dpr;
    cv.height = 120 * dpr;
    cv.style.height = '120px';
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, 120);

    const losses = history.map(h => h.loss);
    const accs = history.map(h => h.accuracy);
    const hasAcc = accs.some(a => a > 0);  // 回归任务无准确率曲线
    const maxL = Math.max(...losses), minL = Math.min(...losses);
    const pad = 6;
    const denom = Math.max(1, losses.length - 1);  // 单点时防除零

    // 损失曲线（橙）
    c.beginPath();
    losses.forEach((l, i) => {
        const x = pad + (w - pad * 2) * i / denom;
        const y = pad + (108 - pad) * (1 - (l - minL) / (maxL - minL + 1e-9));
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    });
    c.strokeStyle = '#e6850f';
    c.lineWidth = 2;
    c.stroke();

    // 准确率曲线（蓝，仅分类任务）
    if (hasAcc) {
        c.beginPath();
        accs.forEach((a, i) => {
            const x = pad + (w - pad * 2) * i / denom;
            const y = pad + (108 - pad) * (1 - a);
            i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
        });
        c.strokeStyle = '#1a73e8';
        c.lineWidth = 2;
        c.stroke();
    }

    // 图例
    c.font = '10px sans-serif';
    c.fillStyle = '#e6850f';
    c.fillText(`损失 ${losses[losses.length - 1].toFixed(3)}`, 8, 12);
    if (hasAcc) {
        c.fillStyle = '#1a73e8';
        c.fillText(`准确率 ${(accs[accs.length - 1] * 100).toFixed(1)}%`, 90, 12);
    }
}

// ===================== 动画控制 =====================
// 每帧基础间隔（ms）：放慢节奏，让演化过程看得清
const BASE_INTERVAL = { cluster: 600, embed: 400, grid: 350, reg: 350 };
const SPEED_OPTIONS = [1, 0.5, 2];  // 播放速度档位循环：1× → 0.5× → 2×

function playAnim() {
    if (state.anim.playing) return;
    state.anim.playing = true;
    $('anim-play').textContent = '⏸';
    $('btn-play').textContent = '⏸ 暂停';
    const base = BASE_INTERVAL[state.anim.type] || 350;
    const interval = Math.round(base / state.anim.speed);
    state.anim.timer = setInterval(() => {
        state.anim.frame++;
        if (state.anim.frame >= state.anim.frames.length - 1) {
            state.anim.frame = state.anim.frames.length - 1;
            stopAnim();
        }
        draw();
        // 分类/回归动画同步损失曲线进度
        if ((state.anim.type === 'grid' || state.anim.type === 'reg')
            && state.result.history && state.result.history.length > 1) {
            const hist = state.result.history, frames = state.result.frames;
            // history与帧等长时按帧索引；不等长（如MLP逐epoch记录）按帧内epoch对齐
            const upto = hist.length === frames.length
                ? state.anim.frame + 2
                : (frames[state.anim.frame].epoch ?? state.anim.frame) + 1;
            drawLossChart(hist.slice(0, upto));
        }
        // GMM聚类：同步对数似然曲线
        if (state.anim.type === 'cluster'
            && state.result.history && state.result.history.length > 1) {
            drawLossChart(state.result.history.slice(0, state.anim.frame + 1));
        }
    }, interval);
}

function stopAnim() {
    state.anim.playing = false;
    $('anim-play').textContent = '▶';
    $('btn-play').textContent = '▶ 开始播放';
    if (state.anim.timer) { clearInterval(state.anim.timer); state.anim.timer = null; }
}

function toggleAnim() {
    if (state.anim.playing) stopAnim();
    else {
        // 已到末尾则回到初始未执行状态重新播放
        if (state.anim.frame >= state.anim.frames.length - 1) {
            state.anim.frame = 0;
            draw();
            if (state.result.history && state.result.history.length > 1)
                drawLossChart(state.result.history.slice(0, 1));
        }
        playAnim();
    }
}

function cycleSpeed() {
    const idx = SPEED_OPTIONS.indexOf(state.anim.speed);
    state.anim.speed = SPEED_OPTIONS[(idx + 1) % SPEED_OPTIONS.length];
    $('anim-speed').textContent = state.anim.speed + '×';
    // 播放中切换速度：重启定时器使其立即生效
    if (state.anim.playing) {
        clearInterval(state.anim.timer);
        state.anim.timer = null;
        state.anim.playing = false;
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
            for (const [child, label] of [[node.left, '≤'], [node.right, '>']]) {
                const path = document.createElementNS(svgNS, 'path');
                const x1 = node._x + NODE_W / 2, y1 = node._y + NODE_H;
                const x2 = child._x + NODE_W / 2, y2 = child._y;
                const my = (y1 + y2) / 2;
                path.setAttribute('d', `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`);
                path.setAttribute('class', 'tedge');
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
        rect.setAttribute('rx', 5);
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

// ===================== GitHub Star 数 =====================
async function loadStarCount() {
    try {
        const res = await fetch('https://api.github.com/repos/zanezhao0708/DataMiningAlgorithm');
        if (!res.ok) return;
        const data = await res.json();
        const el = $('star-count');
        if (el && typeof data.stargazers_count === 'number') {
            el.textContent = data.stargazers_count >= 1000
                ? (data.stargazers_count / 1000).toFixed(1) + 'k'
                : data.stargazers_count;
        }
    } catch (e) {
        // 网络受限（如离线环境）时静默忽略，按钮仍可点击跳转
    }
}

// ===================== 启动 =====================
// resize 防抖：避免拖拽窗口时高频重绘 50×50 决策边界网格
let _resizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(draw, 120);
});
init();
loadStarCount();
