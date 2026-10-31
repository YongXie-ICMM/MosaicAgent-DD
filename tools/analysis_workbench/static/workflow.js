(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const stepIds = ['scan', 'logs', 'colour', 'stitch', 'review', 'layers', 'spatial'];
  const kindNames = { stitch: '自动校色与拼接', layers: '层数识别', spatial: 'Figure 4 空间图', handover: '日志交接' };
  const statusNames = { prepared: '参数已准备', starting: '正在启动', running: '正在处理', completed: '已核验输出', failed: '运行失败', attention: '需要核查', cancelled: '已取消' };
  const activeStatuses = new Set(['starting', 'running']);
  let appState = null;
  let workflowState = { prepared: [], jobs: [], handover: null };
  let connected = false;
  let pending = null;
  let refreshPromise = null;
  let activeStep = 'scan';
  let selectedTicket = '';
  let manuallySelectedTicket = false;
  let handoverEdited = false;
  let pollTimer = null;
  const preferencesKey = 'mosaic-dd-workflow-paths-v1';
  const persistedInputs = ['handoverPath', 'configPath', 'maskPath', 'sampleName', 'validPath', 'exclusionPath'];

  function localized(value) {
    if (typeof value === 'string') return value;
    if (value && typeof value === 'object') return value.zh || value.en || '';
    return '';
  }

  function safeUrl(value) {
    if (typeof value !== 'string' || !value || /[\u0000-\u001f]/.test(value)) return null;
    try {
      const url = new URL(value, window.location.origin);
      if (url.origin !== window.location.origin || !['http:', 'https:'].includes(url.protocol)) return null;
      return url.href;
    } catch (_) { return null; }
  }

  function basename(value) { return typeof value === 'string' ? value.split(/[\\/]/).filter(Boolean).pop() || value : ''; }
  function normalizedPath(value) { const path = typeof value === 'string' ? value.trim().replace(/\\/g, '/').replace(/\/+$/, '') : ''; return /^[A-Za-z]:\//.test(path) ? path.toLowerCase() : path; }
  function batchChanged() {
    const entered = normalizedPath($('handoverPath').value);
    const checked = normalizedPath(workflowState.handover && workflowState.handover.path);
    return entered ? entered !== checked : handoverEdited && !!checked;
  }
  function ordered(records) { return [...records].sort((a, b) => (Date.parse(a.created_at || '') || 0) - (Date.parse(b.created_at || '') || 0)); }
  function jobs() { return ordered(Array.isArray(workflowState.jobs) ? workflowState.jobs : []); }
  function currentJobs() {
    if (batchChanged()) return [];
    if (Array.isArray(workflowState.active_jobs)) return ordered(workflowState.active_jobs);
    const path = normalizedPath(workflowState.handover && workflowState.handover.path);
    return path ? jobs().filter(job => normalizedPath(job.source_dir) === path) : [];
  }
  function prepared() {
    if (batchChanged()) return [];
    let choices = Array.isArray(workflowState.active_prepared) ? workflowState.active_prepared : Array.isArray(workflowState.prepared) ? workflowState.prepared : [];
    const path = normalizedPath(workflowState.handover && workflowState.handover.path);
    if (!Array.isArray(workflowState.active_prepared) && path) choices = choices.filter(item => normalizedPath(item.source_dir) === path);
    return ordered(choices.filter(item => item && item.kind === 'stitch' && item.id));
  }
  function latest(kind) { return [...currentJobs()].reverse().find(item => item.kind === kind) || null; }
  function activeJob() { return jobs().find(item => item.id === workflowState.active_job_id && activeStatuses.has(item.status)) || [...jobs()].reverse().find(item => activeStatuses.has(item.status)) || null; }
  function ticket() { return prepared().find(item => item.id === selectedTicket) || null; }

  function artifacts(job) {
    if (!job) return [];
    let entries = job.artifacts || [];
    if (!Array.isArray(entries) && entries && typeof entries === 'object') {
      entries = Object.entries(entries).map(([name, item]) => typeof item === 'string' ? { name, url: item } : { name, ...item });
    }
    if (!Array.isArray(entries)) return [];
    return entries.filter(item => item && typeof item === 'object' && safeUrl(item.url)).map(item => ({ ...item, url: safeUrl(item.url), name: localized(item.name || item.label) || basename(item.path || item.url) }));
  }

  function imageArtifact(job) {
    const entries = artifacts(job).filter(item => ['image', 'preview'].includes(item.type || item.kind || item.role) || /\.(png|jpe?g|webp)(?:$|[?#])/i.test(item.name || item.url));
    return entries.find(item => /mosaic|拼接/i.test(item.name)) || entries.find(item => item.role === 'preview') || entries[0] || null;
  }

  function notice(message, error = false) {
    $('notice').textContent = message || '';
    $('notice').className = `notice${error ? ' error' : ''}`;
    $('notice').hidden = !message;
  }

  async function request(path, payload) {
    const options = payload === undefined ? { cache: 'no-store' } : {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Workbench-Token': appState && appState.csrf_token || '' }, body: JSON.stringify(payload)
    };
    const response = await fetch(path, options);
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`工作台未返回可读结果（HTTP ${response.status}）。`); }
    if (!response.ok || result.ok === false) throw new Error(localized(result.message) || localized(result.error) || `操作未完成（HTTP ${response.status}）。`);
    return result;
  }

  function remember() {
    const values = {};
    persistedInputs.forEach(id => { values[id] = $(id).value; });
    try { localStorage.setItem(preferencesKey, JSON.stringify(values)); } catch (_) { /* Optional local path preferences only. */ }
  }

  function restore() {
    try {
      const values = JSON.parse(localStorage.getItem(preferencesKey) || '{}');
      persistedInputs.forEach(id => { if (typeof values[id] === 'string') $(id).value = values[id]; });
    } catch (_) { /* Corrupt preferences never prevent a workflow run. */ }
    const requested = new URLSearchParams(window.location.search).get('step');
    if (stepIds.includes(requested)) activeStep = requested;
  }

  function selectStep(id, focus = false) {
    if (!stepIds.includes(id)) return;
    if (id !== activeStep) notice('');
    activeStep = id;
    renderTabs();
    if (focus) {
      const heading = $(`panel-${id}`).querySelector('h2');
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
  }

  function renderTabs() {
    document.querySelectorAll('[data-step]').forEach(button => {
      const selected = button.dataset.step === activeStep;
      button.classList.toggle('active', selected);
      if (selected) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
      $(`panel-${button.dataset.step}`).hidden = !selected;
    });
    const index = stepIds.indexOf(activeStep);
    $('previousStep').disabled = index === 0;
    $('nextStep').disabled = index === stepIds.length - 1;
  }

  function setStepStatus(id, text, state = '') {
    const element = document.querySelector(`[data-status="${id}"]`);
    element.textContent = text;
    const button = element.closest('[data-step]');
    ['done', 'running', 'failed'].forEach(name => button.classList.toggle(name, name === state));
  }

  function renderStatuses() {
    const scannerReady = workflowState.scanner && workflowState.scanner.dd_communication_ready === true;
    setStepStatus('scan', !scannerReady ? '先连接通信日志版' : appState && appState.platform !== 'Windows' ? '在仪器电脑采集' : '可打开采集工具');
    const handover = workflowState.handover;
    const handoverOk = handover && ['verified', 'completed', 'ready'].includes(handover.status);
    if (handoverOk && !batchChanged()) setStepStatus('scan', '采集记录已完成', 'done');
    setStepStatus('logs', batchChanged() ? '新批次等待检查' : handoverOk ? '记录已核验' : handover && handover.status === 'failed' ? '需要检查' : '等待检查', !batchChanged() && handoverOk ? 'done' : !batchChanged() && handover && handover.status === 'failed' ? 'failed' : '');
    const stitch = latest('stitch');
    const colour = stitch && (stitch.colour_status || stitch.colour_correction && stitch.colour_correction.status);
    const colourVerified = stitch && (stitch.render_verified === true || stitch.colour_render_verified === true || stitch.colour_correction && stitch.colour_correction.render_verified === true);
    const actualColourDone = colourVerified && ['applied_at_render', 'verified_prior_correction'].includes(colour);
    const stage = stitch && stitch.progress && stitch.progress.stage || '';
    setStepStatus('colour', actualColourDone ? '校色输出已核验' : colour === 'failed' || colour === 'render_failed' ? '需要检查' : activeStatuses.has(stitch && stitch.status) && /colour|color|校色/.test(stage) ? '正在自动校色' : '随拼接自动执行', actualColourDone ? 'done' : ['failed', 'render_failed'].includes(colour) ? 'failed' : activeStatuses.has(stitch && stitch.status) ? 'running' : '');
    setStepStatus('stitch', stitch ? statusNames[stitch.status] || '等待处理' : prepared().some(item => item.status === 'prepared') ? '参数已准备' : '等待准备', stitch && stitch.status === 'completed' ? 'done' : stitch && ['failed', 'attention'].includes(stitch.status) ? 'failed' : activeStatuses.has(stitch && stitch.status) ? 'running' : '');
    setStepStatus('review', stitch && stitch.status === 'completed' ? '结果可复核' : '等待结果');
    ['layers', 'spatial'].forEach(kind => {
      const job = latest(kind);
      setStepStatus(kind, job ? statusNames[job.status] || '等待处理' : kind === 'layers' ? '等待分析' : '等待层数图', job && job.status === 'completed' ? 'done' : job && ['failed', 'attention'].includes(job.status) ? 'failed' : activeStatuses.has(job && job.status) ? 'running' : '');
    });
  }

  function renderPrepared() {
    const choices = prepared();
    if (!manuallySelectedTicket || !choices.some(item => item.id === selectedTicket)) selectedTicket = choices.length ? choices[choices.length - 1].id : '';
    const select = $('preparedSelect');
    const previous = select.value;
    const signature = choices.map(item => `${item.id}:${item.status}:${item.bundle_dir || item.output_dir || ''}`).join('|');
    if (select.dataset.signature !== signature) {
      select.replaceChildren();
      if (!choices.length) {
        const option = document.createElement('option'); option.value = ''; option.textContent = '暂无已准备批次'; select.appendChild(option);
      }
      choices.forEach(item => {
        const option = document.createElement('option'); option.value = item.id; option.textContent = `${basename(item.bundle_dir || item.output_dir) || item.id}${item.status !== 'prepared' ? '（已提交）' : ''}`; select.appendChild(option);
      });
      select.dataset.signature = signature;
    }
    select.value = selectedTicket || previous;
    const selected = ticket();
    $('preparedName').textContent = selected ? basename(selected.bundle_dir || selected.output_dir) || '已准备的拼接设置' : '尚未准备拼接参数';
    $('preparedPath').textContent = selected ? selected.bundle_dir || selected.output_dir || selected.id : '先到“自动校色”步骤检查原图并准备参数。';
    $('preparedDetails').hidden = choices.length < 2;
  }

  function appendArtifactLinks(container, job) {
    container.replaceChildren();
    artifacts(job).forEach(item => {
      const link = document.createElement('a'); link.href = item.url; link.textContent = `${item.name} ↗`; link.target = '_blank'; link.rel = 'noopener noreferrer'; container.appendChild(link);
    });
  }

  function jobDescription(job) {
    if (!job) return '';
    const parts = [statusNames[job.status] || '已记录'];
    if (activeStatuses.has(job.status)) parts.push(progressDetail(job));
    if (job.error) parts.push(localized(job.error) || String(job.error));
    if (job.output_dir) parts.push(`输出：${job.output_dir}`);
    return parts.join(' · ');
  }

  function progressDetail(job) {
    const text = job.progress && localized(job.progress.detail);
    if (text && text !== 'Intermediate outputs await final source and result validation.') return text;
    const steps = job.progress && Array.isArray(job.progress.completed_steps) ? job.progress.completed_steps : [];
    const names = { qc: '图像质量检查', conflict: '重复照片核对', register: '图像配准', rescue: '完整扫描范围检查', colour_correction: '校色输出核验', render: '拼接出图', summary: '结果汇总' };
    const recorded = steps.length ? names[steps[steps.length - 1]] : '';
    if (job.colour_status === 'ready_for_render') return '已估计校色增益，等待拼接出图与核验。';
    return recorded ? `最近已记录：${recorded}；后台继续处理中。` : '后台正在处理，完成后核验来源与输出文件。';
  }

  function renderProgress(id, job) {
    const element = $(id);
    element.hidden = !job;
    element.textContent = jobDescription(job);
    element.className = `progress-note${job ? ' ' + (['starting', 'running'].includes(job.status) ? 'running' : ['failed', 'attention'].includes(job.status) ? 'failed' : job.status === 'completed' ? 'completed' : '') : ''}`;
  }

  function renderJobs() {
    const list = $('jobs'); list.replaceChildren();
    const records = [...jobs()].reverse().slice(0, 8);
    $('jobCount').textContent = String(jobs().length);
    if (!records.length) {
      const empty = document.createElement('li'); empty.className = 'empty-job'; empty.textContent = '启动的任务和实际结果会显示在这里。'; list.appendChild(empty);
    }
    records.forEach(job => {
      const item = document.createElement('li');
      item.className = ['starting', 'running', 'completed', 'failed', 'attention'].includes(job.status) ? job.status : '';
      const time = document.createElement('time');
      const dateText = job.started_at || job.created_at || job.finished_at;
      const date = dateText ? new Date(dateText) : null;
      time.textContent = date && !Number.isNaN(date.valueOf()) ? date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '已记录';
      const sourceLabel = basename(job.source_dir || job.source_binding && job.source_binding.sources && job.source_binding.sources.mask && job.source_binding.sources.mask.path || '');
      const heading = document.createElement('strong'); heading.textContent = `${kindNames[job.kind] || '分析任务'} · ${statusNames[job.status] || '已记录'}${sourceLabel ? ' · ' + sourceLabel : ''}`;
      const detail = document.createElement('p'); detail.textContent = localized(job.error) || job.error && String(job.error) || activeStatuses.has(job.status) && progressDetail(job) || job.output_dir || '';
      item.append(time, heading, detail);
      if (typeof job.id === 'string' && job.id) { const link = document.createElement('a'); link.href = `/api/workflow/log?job=${encodeURIComponent(job.id)}`; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = '查看运行日志 ↗'; item.appendChild(link); }
      list.appendChild(item);
    });
  }

  function renderReview() {
    const stitch = latest('stitch');
    const image = stitch && stitch.status === 'completed' ? imageArtifact(stitch) : null;
    const open = $('openMosaic');
    open.classList.toggle('disabled', !image);
    open.setAttribute('aria-disabled', String(!image));
    if (image) { open.href = image.url; open.target = '_blank'; open.rel = 'noopener noreferrer'; open.removeAttribute('tabindex'); }
    else { open.removeAttribute('href'); open.setAttribute('tabindex', '-1'); }
    const preview = $('mosaicImage');
    if (image && preview.dataset.url !== image.url) {
      preview.dataset.url = image.url; preview.src = image.url;
      preview.hidden = false; $('mosaicEmpty').hidden = true;
    } else if (!image) { preview.hidden = true; preview.removeAttribute('src'); delete preview.dataset.url; $('mosaicEmpty').hidden = false; }
    $('mosaicCaption').textContent = image ? image.name : stitch && stitch.status === 'completed' ? '输出已核验；此任务未提供可直接预览的图片链接。' : '尚无可用预览';
    appendArtifactLinks($('stitchArtifacts'), stitch);
    appendArtifactLinks($('spatialArtifacts'), latest('spatial'));
  }

  function render() {
    const layer = latest('layers');
    if (layer && layer.status === 'completed' && Array.isArray(layer.samples) && layer.samples.length === 1) {
      const sample = layer.samples[0];
      if (!$('maskPath').value && sample.mask) {
        $('maskPath').value = sample.mask;
        $('validPath').value = sample.valid_mask || '';
        $('exclusionPath').value = sample.exclusion_mask || '';
        $('sampleName').value = sample.sample || '';
      }
    }
    $('connection').className = `connection ${connected ? 'connected' : 'offline'}`;
    $('connection').textContent = connected ? '本机工作台已连接' : '本机连接暂不可用';
    const project = appState && appState.project;
    $('projectName').textContent = project && project.root ? project.name || basename(project.root) : '尚未连接项目';
    $('projectPath').textContent = project && project.root || '在“项目与仪器设置”中连接数据目录。';
    $('projectName').title = project && project.root || '';
    const scannerReady = workflowState.scanner && workflowState.scanner.dd_communication_ready === true;
    const platformNote = appState && appState.platform !== 'Windows' ? '当前电脑用于分析。显微扫描请在连接相机与位移台的 Windows 仪器电脑上进行。' : '';
    $('scanPlatform').textContent = !scannerReady ? `${platformNote}老师先连接通信日志版扫描程序。` : platformNote || '已连接经过版本核验的通信日志版扫描程序；采集参数仍在原有扫描窗口中设置。';
    const handover = workflowState.handover;
    if (!handoverEdited && !$('handoverPath').value && handover && handover.path) $('handoverPath').value = handover.path;
    const handoverLabel = !handover || handover.status === 'waiting' ? '尚未检查' : ['verified', 'completed', 'ready'].includes(handover.status) ? `已核验${Number.isInteger(handover.photos) ? ' ' + handover.photos + ' 张照片及对应采集记录' : '采集记录'}` : handover.status === 'failed' ? '检查未通过' : localized(handover.message) || '已有检查记录';
    $('handoverStatus').textContent = batchChanged() ? '新路径尚未核验，请先检查本批次日志。' : `${handoverLabel}${handover && handover.path ? ' · ' + handover.path : ''}`;
    $('handoverResult').hidden = batchChanged() || !handover || handover.status === 'waiting';
    $('handoverResult').textContent = handover ? localized(handover.message) || `${handoverLabel}${handover.path ? '：' + handover.path : ''}` : '';
    $('handoverResult').className = `result-note${handover && ['verified', 'completed', 'ready'].includes(handover.status) ? ' completed' : handover && handover.status === 'failed' ? ' failed' : ''}`;
    const active = activeJob();
    const batchActive = currentJobs().find(job => activeStatuses.has(job.status));
    $('activeStatus').textContent = batchActive ? `${kindNames[batchActive.kind] || '分析任务'} · ${statusNames[batchActive.status] || batchActive.status}` : active ? '另一批次正在处理，可在运行记录中查看。' : currentJobs().length ? '当前没有运行中的任务' : '本批次尚未运行';
    const memory = workflowState.memory || {};
    $('memoryPath').textContent = memory.root || workflowState.memory_root || '等待读取 Memory 目录。';
    $('memoryCount').textContent = Number.isInteger(memory.event_count) ? `${memory.event_count} 条主记录 · ${memory.workflow_recent_event_count || 0} 条近期步骤记录` : '等待记录';
    renderPrepared(); renderStatuses(); renderJobs(); renderReview(); renderTabs();
    renderProgress('stitchProgress', latest('stitch')); renderProgress('layersProgress', latest('layers')); renderProgress('spatialProgress', latest('spatial'));
    const locked = !connected || !!pending || !!active || batchChanged();
    $('launchScan').disabled = !connected || !!pending || !scannerReady || !appState || appState.platform !== 'Windows';
    $('checkHandover').disabled = !connected || !!pending;
    $('startStitch').disabled = locked || !ticket() || ticket().status !== 'prepared';
    $('startLayers').disabled = locked;
    $('startSpatial').disabled = locked;
    const labels = { launchScan: '打开扫描程序 ↗', checkHandover: '检查交接数据与日志 →', startStitch: '开始自动校色与拼接 →', startLayers: '运行层数识别 →', startSpatial: '生成 Figure 4 空间分布图 →' };
    Object.entries(labels).forEach(([id, label]) => { $(id).textContent = pending === id ? '正在提交…' : label; });
    if (!pending && ticket() && ticket().status !== 'prepared') $('startStitch').textContent = '本批次已提交 · 查看运行记录';
  }

  async function refresh(manual = false) {
    if (refreshPromise) return refreshPromise;
    refreshPromise = (async () => {
      try {
        const [app, result] = await Promise.all([request('/api/state'), request('/api/workflow/state')]);
        appState = app;
        workflowState = result.workflow || result.state || result;
        connected = true;
        render();
      } catch (error) {
        connected = false; render();
        if (manual || !appState) notice(error.message || '请确认本机工作台启动窗口仍在运行，然后刷新。', true);
      } finally { refreshPromise = null; }
    })();
    return refreshPromise;
  }

  async function perform(buttonId, path, body, message) {
    if (!connected || pending) return;
    pending = buttonId; notice(''); render(); remember();
    try {
      const result = await request(path, body);
      notice(localized(result.message) || message);
      if (buttonId === 'launchScan' && safeUrl(result.url)) window.open(safeUrl(result.url), '_blank', 'noopener');
    } catch (error) { notice(error.message || '操作未完成，请查看运行记录。', true); }
    finally { pending = null; await refresh(); render(); }
  }

  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => selectStep(button.dataset.step)));
  $('previousStep').addEventListener('click', () => selectStep(stepIds[stepIds.indexOf(activeStep) - 1], true));
  $('nextStep').addEventListener('click', () => selectStep(stepIds[stepIds.indexOf(activeStep) + 1], true));
  $('refresh').addEventListener('click', () => refresh(true));
  persistedInputs.forEach(id => $(id).addEventListener('change', remember));
  $('handoverPath').addEventListener('input', () => { handoverEdited = true; manuallySelectedTicket = false; render(); });
  $('preparedSelect').addEventListener('change', () => { selectedTicket = $('preparedSelect').value; manuallySelectedTicket = true; render(); });
  $('launchScan').addEventListener('click', () => perform('launchScan', '/api/workflow/scanner', {}, '已请求打开通信日志版采集工具。请在采集窗口查看连接状态。'));
  $('handoverForm').addEventListener('submit', event => { event.preventDefault(); perform('checkHandover', '/api/workflow/handover', { path: $('handoverPath').value.trim() }, '检查请求已返回；核验结果见本批次记录。'); });
  $('startStitch').addEventListener('click', () => { const selected = ticket(); if (selected) perform('startStitch', '/api/workflow/stitch', { ticket: selected.id }, '任务已提交。后台运行及输出核验结果会显示在这里。'); });
  $('layersForm').addEventListener('submit', event => { event.preventDefault(); perform('startLayers', '/api/workflow/layers', { config_path: $('configPath').value.trim() }, '层数分析任务已提交。请查看实际运行记录。'); });
  $('spatialForm').addEventListener('submit', event => {
    event.preventDefault();
    const valid = $('validPath').value.trim();
    if (!valid && !$('wholeCanvas').checked) {
      $('supportDetails').open = true;
      notice('请提供有效扫描区域掩膜；只有在整个画布均有真实扫描覆盖时，才勾选“整个画布有效”。', true);
      $('validPath').focus(); return;
    }
    perform('startSpatial', '/api/workflow/spatial', { mask_path: $('maskPath').value.trim(), valid_path: valid, exclusion_path: $('exclusionPath').value.trim(), sample: $('sampleName').value.trim(), whole_canvas_supported: !valid && $('wholeCanvas').checked }, '空间统计任务已提交。生成的 PNG、SVG 和数据会显示在结果链接中。');
  });
  $('maskPath').addEventListener('change', () => { $('wholeCanvas').checked = false; });
  $('configPath').addEventListener('invalid', () => { $('configDetails').open = true; });
  $('mosaicImage').addEventListener('error', () => { $('mosaicImage').hidden = true; $('mosaicEmpty').hidden = false; $('mosaicCaption').textContent = '预览未能载入，请通过下方结果文件链接核查。'; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  restore(); renderTabs(); refresh();
  pollTimer = window.setInterval(() => { if (!document.hidden) refresh(); }, 5000);
  window.addEventListener('pagehide', () => { if (pollTimer) window.clearInterval(pollTimer); pollTimer = null; });
  window.addEventListener('pageshow', () => {
    if (!pollTimer) pollTimer = window.setInterval(() => { if (!document.hidden) refresh(); }, 5000);
    refresh();
  });
})();
