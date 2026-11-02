(() => {
  'use strict';

  const i18n = window.DDWorkflowI18n;
  const t = i18n.t;
  const $ = id => document.getElementById(id);
  const stepIds = ['scan', 'logs', 'colour', 'stitch', 'review', 'layers', 'spatial'];
  const kindNames = () => ({ stitch: t('自动校色与拼接'), layers: t('层数识别'), spatial: t('Figure 4 空间图'), handover: t('日志交接') });
  const statusNames = () => ({ prepared: t('参数已准备'), starting: t('正在启动'), running: t('正在处理'), completed: t('已核验输出'), failed: t('运行失败'), attention: t('需要核查'), cancelled: t('已取消') });
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
  let lastNotice = null;
  let spatialReview = null;
  let rectangleDrag = null;
  let spatialFieldsEdited = false;
  const preferencesKey = 'mosaic-dd-workflow-paths-v1';
  const persistedInputs = ['handoverPath', 'configPath', 'maskPath', 'sampleName', 'validPath', 'exclusionPath', 'spatialMosaicPath'];

  function localized(value) { return i18n.localized(value); }

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
  function spatialJobMatchesInputs(job) {
    if (spatialReview && (job.id !== spatialReview.submittedJobId || JSON.stringify(spatialReview.rectangles) !== spatialReview.submittedRectangles)) return false;
    const binding = job.source_binding || {};
    const sources = binding.review_sources || binding.sources || {};
    const expected = { mask: $('maskPath').value, mosaic: $('spatialMosaicPath').value, valid_support: $('validPath').value, exclusions: $('exclusionPath').value };
    if (!normalizedPath(expected.mask)) return false;
    if (!normalizedPath(expected.valid_support) && binding.whole_canvas_supported !== $('wholeCanvas').checked) return false;
    return Object.entries(expected).every(([key, path]) => normalizedPath(path) === normalizedPath(sources[key] && sources[key].path));
  }
  function latest(kind) { return [...currentJobs()].reverse().find(item => item.kind === kind && (kind !== 'spatial' || spatialJobMatchesInputs(item))) || null; }
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
    lastNotice = { message, error };
    $('notice').textContent = localized(message) || '';
    $('notice').className = `notice${error ? ' error' : ''}`;
    $('notice').hidden = !message;
  }

  async function request(path, payload) {
    const options = payload === undefined ? { cache: 'no-store' } : {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Workbench-Token': appState && appState.csrf_token || '' }, body: JSON.stringify(payload)
    };
    const response = await fetch(path, options);
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(t('工作台未返回可读结果（HTTP {status}）。', { status: response.status })); }
    if (!response.ok || result.ok === false) throw new Error(localized(result.message) || localized(result.error) || t('操作未完成（HTTP {status}）。', { status: response.status }));
    return result;
  }

  function remember() {
    const values = {};
    persistedInputs.forEach(id => { values[id] = $(id).value; });
    values.autoCentralExclusion = $('autoCentralExclusion').checked;
    try { localStorage.setItem(preferencesKey, JSON.stringify(values)); } catch (_) { /* Optional local path preferences only. */ }
  }

  function restore() {
    try {
      const values = JSON.parse(localStorage.getItem(preferencesKey) || '{}');
      persistedInputs.forEach(id => { if (typeof values[id] === 'string') $(id).value = values[id]; });
      if (typeof values.autoCentralExclusion === 'boolean') $('autoCentralExclusion').checked = values.autoCentralExclusion;
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
    setStepStatus('scan', !scannerReady ? t('先连接通信日志版') : appState && appState.platform !== 'Windows' ? t('在仪器电脑采集') : t('可打开采集工具'));
    const handover = workflowState.handover;
    const handoverOk = handover && ['verified', 'completed', 'ready'].includes(handover.status);
    if (handoverOk && !batchChanged()) setStepStatus('scan', t('采集记录已完成'), 'done');
    setStepStatus('logs', batchChanged() ? t('新批次等待检查') : handoverOk ? t('记录已核验') : handover && handover.status === 'failed' ? t('需要检查') : t('等待检查'), !batchChanged() && handoverOk ? 'done' : !batchChanged() && handover && handover.status === 'failed' ? 'failed' : '');
    const stitch = latest('stitch');
    const colour = stitch && (stitch.colour_status || stitch.colour_correction && stitch.colour_correction.status);
    const colourVerified = stitch && (stitch.render_verified === true || stitch.colour_render_verified === true || stitch.colour_correction && stitch.colour_correction.render_verified === true);
    const actualColourDone = colourVerified && ['applied_at_render', 'verified_prior_correction'].includes(colour);
    const stage = stitch && stitch.progress && stitch.progress.stage || '';
    setStepStatus('colour', actualColourDone ? t('校色输出已核验') : colour === 'failed' || colour === 'render_failed' ? t('需要检查') : activeStatuses.has(stitch && stitch.status) && /colour|color|校色/.test(stage) ? t('正在自动校色') : t('随拼接自动执行'), actualColourDone ? 'done' : ['failed', 'render_failed'].includes(colour) ? 'failed' : activeStatuses.has(stitch && stitch.status) ? 'running' : '');
    setStepStatus('stitch', stitch ? statusNames()[stitch.status] || t('等待处理') : prepared().some(item => item.status === 'prepared') ? t('参数已准备') : t('等待准备'), stitch && stitch.status === 'completed' ? 'done' : stitch && ['failed', 'attention'].includes(stitch.status) ? 'failed' : activeStatuses.has(stitch && stitch.status) ? 'running' : '');
    setStepStatus('review', stitch && stitch.status === 'completed' ? t('结果可复核') : t('等待结果'));
    ['layers', 'spatial'].forEach(kind => {
      const job = latest(kind);
      const waiting = kind === 'layers' ? t('等待分析') : spatialReview ? t('等待人工确认') : $('maskPath').value.trim() ? t('准备排除框复核') : t('等待层数图');
      setStepStatus(kind, job ? statusNames()[job.status] || t('等待处理') : waiting, job && job.status === 'completed' ? 'done' : job && ['failed', 'attention'].includes(job.status) ? 'failed' : activeStatuses.has(job && job.status) ? 'running' : '');
    });
  }

  function renderPrepared() {
    const choices = prepared();
    if (!manuallySelectedTicket || !choices.some(item => item.id === selectedTicket)) selectedTicket = choices.length ? choices[choices.length - 1].id : '';
    const select = $('preparedSelect');
    const previous = select.value;
    const signature = i18n.getLanguage() + ':' + choices.map(item => `${item.id}:${item.status}:${item.bundle_dir || item.output_dir || ''}`).join('|');
    if (select.dataset.signature !== signature) {
      select.replaceChildren();
      if (!choices.length) {
        const option = document.createElement('option'); option.value = ''; option.textContent = t('暂无已准备批次'); select.appendChild(option);
      }
      choices.forEach(item => {
        const option = document.createElement('option'); option.value = item.id; option.textContent = `${basename(item.bundle_dir || item.output_dir) || item.id}${item.status !== 'prepared' ? t('（已提交）') : ''}`; select.appendChild(option);
      });
      select.dataset.signature = signature;
    }
    select.value = selectedTicket || previous;
    const selected = ticket();
    $('preparedName').textContent = selected ? basename(selected.bundle_dir || selected.output_dir) || t('已准备的拼接设置') : t('尚未准备拼接参数');
    $('preparedPath').textContent = selected ? selected.bundle_dir || selected.output_dir || selected.id : t('先到“自动校色”步骤检查原图并准备参数。');
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
    const parts = [statusNames()[job.status] || t('已记录')];
    if (activeStatuses.has(job.status)) parts.push(progressDetail(job));
    if (job.error) parts.push(localized(job.error) || String(job.error));
    if (job.output_dir) parts.push(t('输出：{path}', { path: job.output_dir }));
    return parts.join(' · ');
  }

  function progressDetail(job) {
    const text = job.progress && localized(job.progress.detail);
    if (text && text !== 'Intermediate outputs await final source and result validation.') return text;
    const steps = job.progress && Array.isArray(job.progress.completed_steps) ? job.progress.completed_steps : [];
    const names = { qc: t('图像质量检查'), conflict: t('重复照片核对'), register: t('图像配准'), rescue: t('完整扫描范围检查'), colour_correction: t('校色输出核验'), render: t('拼接出图'), summary: t('结果汇总') };
    const recorded = steps.length ? names[steps[steps.length - 1]] : '';
    if (job.colour_status === 'ready_for_render') return t('已估计校色增益，等待拼接出图与核验。');
    return recorded ? t('最近已记录：{step}；后台继续处理中。', { step: recorded }) : t('后台正在处理，完成后核验来源与输出文件。');
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
      const empty = document.createElement('li'); empty.className = 'empty-job'; empty.textContent = t('启动的任务和实际结果会显示在这里。'); list.appendChild(empty);
    }
    records.forEach(job => {
      const item = document.createElement('li');
      item.className = ['starting', 'running', 'completed', 'failed', 'attention'].includes(job.status) ? job.status : '';
      const time = document.createElement('time');
      const dateText = job.started_at || job.created_at || job.finished_at;
      const date = dateText ? new Date(dateText) : null;
      time.textContent = date && !Number.isNaN(date.valueOf()) ? date.toLocaleString(i18n.getLanguage() === 'zh' ? 'zh-CN' : 'en-GB', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : t('已记录');
      const sourceLabel = basename(job.source_dir || job.source_binding && job.source_binding.sources && job.source_binding.sources.mask && job.source_binding.sources.mask.path || '');
      const heading = document.createElement('strong'); heading.textContent = `${kindNames()[job.kind] || t('分析任务')} · ${statusNames()[job.status] || t('已记录')}${sourceLabel ? ' · ' + sourceLabel : ''}`;
      const detail = document.createElement('p'); detail.textContent = localized(job.error) || job.error && String(job.error) || activeStatuses.has(job.status) && progressDetail(job) || job.output_dir || '';
      item.append(time, heading, detail);
      if (typeof job.id === 'string' && job.id) { const link = document.createElement('a'); link.href = `/api/workflow/log?job=${encodeURIComponent(job.id)}`; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = t('查看运行日志 ↗'); item.appendChild(link); }
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
    $('mosaicCaption').textContent = image ? image.name : stitch && stitch.status === 'completed' ? t('输出已核验；此任务未提供可直接预览的图片链接。') : t('尚无可用预览');
    appendArtifactLinks($('stitchArtifacts'), stitch);
    appendArtifactLinks($('spatialArtifacts'), latest('spatial'));
  }

  function spatialInputs() {
    const valid = $('validPath').value.trim();
    const mosaic = $('spatialMosaicPath').value.trim();
    return { mask_path: $('maskPath').value.trim(), valid_path: valid, exclusion_path: $('exclusionPath').value.trim(), sample: $('sampleName').value.trim(), mosaic_path: mosaic, auto_central_exclusion: !!mosaic && $('autoCentralExclusion').checked, whole_canvas_supported: !valid && $('wholeCanvas').checked };
  }

  function invalidateSpatialReview() {
    spatialReview = null; rectangleDrag = null;
    $('confirmReview').checked = false;
    $('spatialReview').hidden = true;
    $('confirmSpatial').disabled = true;
    $('reviewKimiAdvice').textContent = ''; $('reviewKimiAdvice').hidden = true;
  }

  function validRectangle(rectangle, review = spatialReview) {
    return review && ['x1', 'y1', 'x2', 'y2'].every(key => Number.isInteger(rectangle[key])) && rectangle.x1 >= 0 && rectangle.y1 >= 0 && rectangle.x2 <= review.width && rectangle.y2 <= review.height && rectangle.x2 > rectangle.x1 && rectangle.y2 > rectangle.y1;
  }

  function boxesChanged() {
    if (spatialReview) { delete spatialReview.submittedJobId; delete spatialReview.submittedRectangles; }
    $('confirmReview').checked = false;
    $('reviewKimiAdvice').textContent = ''; $('reviewKimiAdvice').hidden = true;
    render();
  }

  function drawReviewCanvas(id, image) {
    const canvas = $(id);
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!image || !spatialReview) return;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const sx = canvas.width / spatialReview.width, sy = canvas.height / spatialReview.height;
    const boxes = spatialReview.rectangles.map((rectangle, index) => ({ rectangle, index }));
    if (rectangleDrag) boxes.push({ rectangle: rectangleDrag.rectangle, index: -1 });
    for (const { rectangle: r, index } of boxes) {
      context.strokeStyle = index === spatialReview.selected ? '#00dbe8' : '#ed3e32';
      context.lineWidth = Math.max(2, canvas.width / 450);
      context.setLineDash(index < 0 ? [7, 4] : []);
      context.strokeRect(r.x1 * sx, r.y1 * sy, (r.x2 - r.x1) * sx, (r.y2 - r.y1) * sy);
      if (index >= 0) {
        context.fillStyle = '#ffffff'; context.font = `${Math.max(13, canvas.width / 60)}px sans-serif`;
        context.fillText(String(index + 1), r.x1 * sx + 4, r.y1 * sy + 17);
      }
    }
    context.setLineDash([]);
  }

  function renderSpatialReview() {
    const review = spatialReview;
    if (review && JSON.stringify(review.inputs) !== JSON.stringify(spatialInputs())) { invalidateSpatialReview(); return; }
    $('spatialReview').hidden = !review;
    if (!review) { $('confirmSpatial').disabled = true; return; }
    const selected = review.rectangles[review.selected];
    const coordinateKey = JSON.stringify([review.selected, selected]);
    if (review.coordinateKey !== coordinateKey) {
      ['x1', 'y1', 'x2', 'y2'].forEach(key => {
        const input = $('rect' + key.toUpperCase());
        input.value = selected ? String(selected[key]) : '';
        input.max = key.startsWith('x') ? review.width : review.height;
      });
      review.coordinateKey = coordinateKey;
      review.coordinateDraft = false;
    }
    $('selectedRectangle').textContent = selected ? t('第 {index} 个框，共 {count} 个', { index: review.selected + 1, count: review.rectangles.length }) : t('尚未选择排除框');
    const status = !review.ready ? t(review.loadFailed ? '预览未能载入，请重新准备复核。' : '预览加载中；还不能确认。') : !review.rectangles.length ? t('没有排除框；确认后会保留所有有效区域（已有排除掩膜仍适用）。') : t('共 {count} 个待确认排除框', { count: review.rectangles.length });
    $('spatialReviewStatus').textContent = status + (review.proposalStatus === 'needs_review' ? ' ' + t('建议框不确定，请手工检查、调整或画框；也可明确确认不排除任何区域。') : '');
    const locked = !connected || !!pending || !!activeJob() || !review.ready;
    ['rectX1', 'rectY1', 'rectX2', 'rectY2', 'applyRectangle', 'deleteRectangle'].forEach(id => { $(id).disabled = locked || !selected; });
    $('clearRectangles').disabled = locked || !review.rectangles.length;
    $('spatialReviewMode').disabled = locked;
    $('confirmReview').disabled = locked;
    $('confirmSpatial').disabled = locked || !!review.coordinateDraft || !$('confirmReview').checked;
    $('reviewKimi').disabled = locked || !review.rectangles.length;
    $('confirmSpatial').textContent = pending === 'confirmSpatial' ? t('正在提交…') : t('确认区域并生成 Figure 4 →');
    $('reviewKimi').textContent = pending === 'reviewKimi' ? t('正在提交…') : t('请 Kimi 核对这些区域');
    drawReviewCanvas('spatialOpticalCanvas', review.optical);
    drawReviewCanvas('spatialMaskCanvas', review.segmentation);
  }

  async function prepareSpatialReview() {
    if (!connected || pending || activeJob()) return;
    const inputs = spatialInputs();
    if (!inputs.mosaic_path) { notice(t('请提供与层数掩膜对应的光学拼接图，用于人工核对排除区域。'), true); $('spatialMosaicPath').focus(); return; }
    if (!inputs.valid_path && !inputs.whole_canvas_supported) {
      $('supportDetails').open = true;
      notice(t('请提供有效扫描区域掩膜；只有在整个画布均有真实扫描覆盖时，才勾选“整个画布有效”。'), true);
      $('validPath').focus(); return;
    }
    invalidateSpatialReview(); pending = 'startSpatial'; notice(''); remember(); render();
    try {
      const result = await request('/api/workflow/spatial-review', inputs);
      const data = result.review || result;
      if (JSON.stringify(inputs) !== JSON.stringify(spatialInputs())) throw new Error(t('输入已改变，请重新准备排除框复核。'));
      if (typeof data.review_id !== 'string' || !data.review_id || !Number.isInteger(data.image_width) || data.image_width <= 0 || !Number.isInteger(data.image_height) || data.image_height <= 0 || !Array.isArray(data.rectangles)) throw new Error(t('复核接口未返回有效的图像尺寸或记录编号。'));
      const automatic = data.automatic_proposal || data.proposal && data.proposal.automatic_proposal || {};
      const review = { id: data.review_id, width: data.image_width, height: data.image_height, rectangles: data.rectangles.map(r => ({ x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2 })), selected: -1, inputs, proposalStatus: automatic.status, ready: false };
      if (!review.rectangles.every(r => validRectangle(r, review))) throw new Error(t('排除框坐标无效，请使用图像范围内的整数，且右边和下边大于左边和上边。'));
      const urls = data.preview_urls || {};
      if (!safeUrl(urls.segmentation)) throw new Error(t('预览未能载入，请重新准备复核。'));
      spatialReview = review;
      $('spatialReviewMode').value = 'draw';
      $('spatialOpticalUnavailable').hidden = !!safeUrl(urls.optical);
      $('spatialOpticalCanvas').hidden = !safeUrl(urls.optical);
      renderSpatialReview();
      await Promise.all(['optical', 'segmentation'].filter(kind => safeUrl(urls[kind])).map(kind => new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
          if (spatialReview === review) {
            review[kind] = image;
            const canvas = $(kind === 'optical' ? 'spatialOpticalCanvas' : 'spatialMaskCanvas');
            canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
            renderSpatialReview();
          }
          resolve();
        };
        image.onerror = () => reject(new Error(t('预览未能载入，请重新准备复核。')));
        image.src = safeUrl(urls[kind]);
      })));
      if (spatialReview === review) review.ready = true;
    } catch (error) {
      if (spatialReview) spatialReview.loadFailed = true;
      notice(error.message || t('操作未完成，请查看运行记录。'), true);
    } finally { pending = null; render(); }
  }

  function reviewPoint(canvas, event) {
    const bounds = canvas.getBoundingClientRect();
    return { x: Math.max(0, Math.min(spatialReview.width, (event.clientX - bounds.left) / bounds.width * spatialReview.width)), y: Math.max(0, Math.min(spatialReview.height, (event.clientY - bounds.top) / bounds.height * spatialReview.height)) };
  }

  function attachRectangleCanvas(id) {
    const canvas = $(id);
    canvas.addEventListener('pointerdown', event => {
      if (!spatialReview || !spatialReview.ready || pending || activeJob()) return;
      const point = reviewPoint(canvas, event);
      if ($('spatialReviewMode').value === 'select') {
        spatialReview.selected = -1;
        spatialReview.rectangles.forEach((r, index) => { if (point.x >= r.x1 && point.x < r.x2 && point.y >= r.y1 && point.y < r.y2) spatialReview.selected = index; });
        renderSpatialReview(); return;
      }
      event.preventDefault();
      rectangleDrag = { canvas: id, start: point, rectangle: { x1: Math.floor(point.x), y1: Math.floor(point.y), x2: Math.ceil(point.x), y2: Math.ceil(point.y) } };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!spatialReview || !rectangleDrag || rectangleDrag.canvas !== id) return;
      const point = reviewPoint(canvas, event), start = rectangleDrag.start;
      rectangleDrag.rectangle = { x1: Math.floor(Math.min(start.x, point.x)), y1: Math.floor(Math.min(start.y, point.y)), x2: Math.ceil(Math.max(start.x, point.x)), y2: Math.ceil(Math.max(start.y, point.y)) };
      renderSpatialReview();
    });
    canvas.addEventListener('pointerup', () => {
      if (!spatialReview || !rectangleDrag || rectangleDrag.canvas !== id) return;
      if (validRectangle(rectangleDrag.rectangle)) {
        spatialReview.rectangles.push(rectangleDrag.rectangle); spatialReview.selected = spatialReview.rectangles.length - 1;
      }
      rectangleDrag = null; boxesChanged();
    });
    canvas.addEventListener('pointercancel', () => { rectangleDrag = null; renderSpatialReview(); });
  }

  function render() {
    const layer = latest('layers');
    if (!spatialFieldsEdited && layer && layer.status === 'completed' && Array.isArray(layer.samples) && layer.samples.length === 1) {
      const sample = layer.samples[0];
      if (!$('maskPath').value && sample.mask) {
        $('maskPath').value = sample.mask;
        $('validPath').value = sample.valid_mask || '';
        $('exclusionPath').value = sample.exclusion_mask || '';
        $('sampleName').value = sample.sample || '';
        $('spatialMosaicPath').value = sample.mosaic || '';
      }
    }
    $('connection').className = `connection ${connected ? 'connected' : 'offline'}`;
    $('connection').textContent = connected ? t('本机工作台已连接') : t('本机连接暂不可用');
    const project = appState && appState.project;
    $('projectName').textContent = project && project.root ? project.name || basename(project.root) : t('尚未连接项目');
    $('projectPath').textContent = project && project.root || t('在“项目与仪器设置”中连接数据目录。');
    $('projectName').title = project && project.root || '';
    const scannerReady = workflowState.scanner && workflowState.scanner.dd_communication_ready === true;
    const platformNote = appState && appState.platform !== 'Windows' ? t('当前电脑用于分析。显微扫描请在连接相机与位移台的 Windows 仪器电脑上进行。') : '';
    $('scanPlatform').textContent = !scannerReady ? platformNote + t('老师先连接通信日志版扫描程序。') : platformNote || t('已连接经过版本核验的通信日志版扫描程序；采集参数仍在原有扫描窗口中设置。');
    const handover = workflowState.handover;
    if (!handoverEdited && !$('handoverPath').value && handover && handover.path) $('handoverPath').value = handover.path;
    const handoverLabel = !handover || handover.status === 'waiting' ? t('尚未检查') : ['verified', 'completed', 'ready'].includes(handover.status) ? (Number.isInteger(handover.photos) ? t('已核验 {count} 张照片及对应采集记录', { count: handover.photos }) : t('已核验采集记录')) : handover.status === 'failed' ? t('检查未通过') : localized(handover.message) || t('已有检查记录');
    $('handoverStatus').textContent = batchChanged() ? t('新路径尚未核验，请先检查本批次日志。') : `${handoverLabel}${handover && handover.path ? ' · ' + handover.path : ''}`;
    $('handoverResult').hidden = batchChanged() || !handover || handover.status === 'waiting';
    $('handoverResult').textContent = handover ? localized(handover.message) || `${handoverLabel}${handover.path ? '：' + handover.path : ''}` : '';
    $('handoverResult').className = `result-note${handover && ['verified', 'completed', 'ready'].includes(handover.status) ? ' completed' : handover && handover.status === 'failed' ? ' failed' : ''}`;
    const active = activeJob();
    const batchActive = currentJobs().find(job => activeStatuses.has(job.status));
    $('activeStatus').textContent = batchActive ? `${kindNames()[batchActive.kind] || t('分析任务')} · ${statusNames()[batchActive.status] || batchActive.status}` : active ? t('另一批次正在处理，可在运行记录中查看。') : currentJobs().length ? t('当前没有运行中的任务') : t('本批次尚未运行');
    const memory = workflowState.memory || {};
    $('memoryPath').textContent = memory.root || workflowState.memory_root || t('等待读取 Memory 目录。');
    $('memoryCount').textContent = Number.isInteger(memory.event_count) ? t('{count} 条主记录 · {recent} 条近期步骤记录', { count: memory.event_count, recent: memory.workflow_recent_event_count || 0 }) : t('等待记录');
    renderPrepared(); renderStatuses(); renderJobs(); renderReview(); renderTabs();
    renderProgress('stitchProgress', latest('stitch')); renderProgress('layersProgress', latest('layers')); renderProgress('spatialProgress', latest('spatial'));
    const locked = !connected || !!pending || !!active || batchChanged();
    $('launchScan').disabled = !connected || !!pending || !scannerReady || !appState || appState.platform !== 'Windows';
    $('checkHandover').disabled = !connected || !!pending;
    $('startStitch').disabled = locked || !ticket() || ticket().status !== 'prepared';
    $('startLayers').disabled = locked;
    $('startSpatial').disabled = locked;
    const labels = { launchScan: t('打开扫描程序 ↗'), checkHandover: t('检查交接数据与日志 →'), startStitch: t('开始自动校色与拼接 →'), startLayers: t('运行层数识别 →'), startSpatial: t('复核排除框 →') };
    Object.entries(labels).forEach(([id, label]) => { $(id).textContent = pending === id ? t('正在提交…') : label; });
    if (!pending && ticket() && ticket().status !== 'prepared') $('startStitch').textContent = t('本批次已提交 · 查看运行记录');
    renderSpatialReview();
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
        if (manual || !appState) notice(error.message || t('请确认本机工作台启动窗口仍在运行，然后刷新。'), true);
      } finally { refreshPromise = null; }
    })();
    return refreshPromise;
  }

  async function perform(buttonId, path, body, message) {
    if (!connected || pending) return;
    let outcome = null;
    pending = buttonId; notice(''); render(); remember();
    try {
      const result = await request(path, body);
      outcome = result;
      notice(result.message || message);
      if (buttonId === 'launchScan' && safeUrl(result.url)) window.open(safeUrl(result.url), '_blank', 'noopener');
    } catch (error) { notice(error.message || t('操作未完成，请查看运行记录。'), true); }
    finally { pending = null; await refresh(); render(); }
    return outcome;
  }

  $('languageSelect').addEventListener('change', () => {
    i18n.setLanguage($('languageSelect').value);
    i18n.apply(document);
    render();
    if (lastNotice) notice(lastNotice.message, lastNotice.error);
  });
  $('autoCentralExclusion').addEventListener('change', remember);
  ['maskPath', 'validPath', 'exclusionPath', 'spatialMosaicPath', 'sampleName', 'wholeCanvas', 'autoCentralExclusion', 'handoverPath'].forEach(id => {
    ['input', 'change'].forEach(event => $(id).addEventListener(event, () => { spatialFieldsEdited = true; invalidateSpatialReview(); render(); }));
  });
  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => selectStep(button.dataset.step)));
  $('previousStep').addEventListener('click', () => selectStep(stepIds[stepIds.indexOf(activeStep) - 1], true));
  $('nextStep').addEventListener('click', () => selectStep(stepIds[stepIds.indexOf(activeStep) + 1], true));
  $('refresh').addEventListener('click', () => refresh(true));
  persistedInputs.forEach(id => $(id).addEventListener('change', remember));
  $('handoverPath').addEventListener('input', () => { handoverEdited = true; manuallySelectedTicket = false; render(); });
  $('preparedSelect').addEventListener('change', () => { selectedTicket = $('preparedSelect').value; manuallySelectedTicket = true; render(); });
  $('launchScan').addEventListener('click', () => perform('launchScan', '/api/workflow/scanner', {}, t('已请求打开通信日志版采集工具。请在采集窗口查看连接状态。')));
  $('handoverForm').addEventListener('submit', event => { event.preventDefault(); perform('checkHandover', '/api/workflow/handover', { path: $('handoverPath').value.trim() }, t('检查请求已返回；核验结果见本批次记录。')); });
  $('startStitch').addEventListener('click', () => { const selected = ticket(); if (selected) perform('startStitch', '/api/workflow/stitch', { ticket: selected.id }, t('任务已提交。后台运行及输出核验结果会显示在这里。')); });
  $('layersForm').addEventListener('submit', event => { event.preventDefault(); perform('startLayers', '/api/workflow/layers', { config_path: $('configPath').value.trim() }, t('层数分析任务已提交。请查看实际运行记录。')); });
  $('spatialForm').addEventListener('submit', event => { event.preventDefault(); prepareSpatialReview(); });
  attachRectangleCanvas('spatialOpticalCanvas'); attachRectangleCanvas('spatialMaskCanvas');
  $('applyRectangle').addEventListener('click', () => {
    if (!spatialReview || pending || spatialReview.selected < 0) return;
    const rectangle = Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map(key => [key, $('rect' + key.toUpperCase()).value.trim() === '' ? NaN : Number($('rect' + key.toUpperCase()).value)]));
    if (!validRectangle(rectangle)) { notice(t('排除框坐标无效，请使用图像范围内的整数，且右边和下边大于左边和上边。'), true); return; }
    spatialReview.rectangles[spatialReview.selected] = rectangle; spatialReview.coordinateDraft = false; boxesChanged();
  });
  ['rectX1', 'rectY1', 'rectX2', 'rectY2'].forEach(id => $(id).addEventListener('input', () => {
    if (spatialReview) { spatialReview.coordinateDraft = true; $('confirmReview').checked = false; $('confirmSpatial').disabled = true; }
  }));
  $('deleteRectangle').addEventListener('click', () => {
    if (!spatialReview || pending || spatialReview.selected < 0) return;
    spatialReview.rectangles.splice(spatialReview.selected, 1); spatialReview.selected = -1; boxesChanged();
  });
  $('clearRectangles').addEventListener('click', () => {
    if (!spatialReview || pending) return;
    spatialReview.rectangles = []; spatialReview.selected = -1; boxesChanged();
  });
  $('confirmReview').addEventListener('change', renderSpatialReview);
  $('confirmSpatial').addEventListener('click', async () => {
    if (!spatialReview || !spatialReview.ready || spatialReview.coordinateDraft || !$('confirmReview').checked || pending) { notice(t('请先载入预览并明确确认排除区域。'), true); return; }
    if (JSON.stringify(spatialInputs()) !== JSON.stringify(spatialReview.inputs)) { invalidateSpatialReview(); notice(t('输入已改变，请重新准备排除框复核。'), true); return; }
    const review = spatialReview;
    const payload = { ...spatialInputs(), auto_central_exclusion: false, review_id: review.id, rectangles: review.rectangles.map(r => ({ ...r })), review_confirmed: true };
    const result = await perform('confirmSpatial', '/api/workflow/spatial', payload, t('已确认的排除区域与空间统计任务已提交。'));
    if (spatialReview === review && result && result.job && result.job.id) { review.submittedJobId = result.job.id; review.submittedRectangles = JSON.stringify(payload.rectangles); }
    $('confirmReview').checked = false; render();
  });
  $('reviewKimi').addEventListener('click', async () => {
    if (!spatialReview || !spatialReview.ready || pending) return;
    if (!spatialReview.rectangles.length) { notice(t('至少画一个排除框后再请求 Kimi 意见。'), true); return; }
    const review = spatialReview; pending = 'reviewKimi'; render();
    try {
      const result = await request('/api/workflow/spatial-review-kimi', { review_id: review.id, rectangles: review.rectangles.map(r => ({ ...r })) });
      if (spatialReview !== review) return;
      const data = result.result || result;
      const advice = data.advice || data.message;
      const details = advice && typeof advice === 'object' && Array.isArray(advice.rectangles) ? advice.rectangles.map(item => `${item.index}. ${item.inclusion || ''} · ${item.reason || ''}`).join('\n') : '';
      $('reviewKimiAdvice').textContent = [advice && typeof advice === 'object' ? localized(advice.summary) || localized(advice) : localized(advice), details, localized(data.reason)].filter(Boolean).join('\n\n') || t('Kimi 未返回可显示的意见。');
      $('reviewKimiAdvice').hidden = false;
    } catch (error) { notice(error.message || t('操作未完成，请查看运行记录。'), true); }
    finally { pending = null; render(); }
  });
  $('maskPath').addEventListener('change', () => { $('wholeCanvas').checked = false; });
  $('configPath').addEventListener('invalid', () => { $('configDetails').open = true; });
  $('mosaicImage').addEventListener('error', () => { $('mosaicImage').hidden = true; $('mosaicEmpty').hidden = false; $('mosaicCaption').textContent = t('预览未能载入，请通过下方结果文件链接核查。'); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  restore(); i18n.apply(document); renderTabs(); refresh();
  pollTimer = window.setInterval(() => { if (!document.hidden) refresh(); }, 5000);
  window.addEventListener('pagehide', () => { if (pollTimer) window.clearInterval(pollTimer); pollTimer = null; });
  window.addEventListener('pageshow', () => {
    if (!pollTimer) pollTimer = window.setInterval(() => { if (!document.hidden) refresh(); }, 5000);
    refresh();
  });
})();
