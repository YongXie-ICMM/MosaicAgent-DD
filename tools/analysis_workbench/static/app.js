(() => {
  'use strict';

  const strings = {
    zh: {
      stitchSetup: '检查新扫描图片与拼接参数 →',
      workbench: '分析工作台', connecting: '正在连接', connected: '本机工作台', disconnected: '连接中断', setup: '设置',
      eyebrow: '从采集到复核', headline: '让每一步分析，都有据可查。', subtitle: '选择数据，查看原图与结果，再决定下一步。',
      currentRun: '当前项目', noRun: '尚未选择数据', ready: '可进入', selectData: '选择数据', optional: '可选',
      scanTitle: '显微扫描', scanDescription: '打开现有 AmScope 采集工具', imagesTitle: '图像与拼接', imagesDescription: '查看原图，检查拼接结果',
      layersTitle: '层数分析', layersDescription: 'U-Net 层数预测与面积统计', spectraTitle: '光谱与位置', spectraDescription: '关联测量，核对坐标与证据',
      launchScanner: '打开 AmScope 扫描工具', scanGuide: '扫描操作说明 ↗', scannerRepository: 'AmScope 仓库 ↗',
      openImages: '打开图像与拼接结果', reviewLayers: '查看层数结果', openSpectra: '打开光谱与位置记录',
      layersHelp: '对照原图核对层数预测，再检查统计区域、有效像素与分母。',
      previewTitle: '先看图，再判断', previewSubtitle: '原图和分析结果并列显示。', showOverlay: '显示叠加', original: '原始图像', result: '分析结果',
      openImage: '打开大图 ↗', originalEmpty: '当前步骤暂无原图', originalEmptyHelp: '选择图像案例，或在设置中连接数据',
      resultEmpty: '当前步骤暂无图像结果', resultEmptyHelp: '只展示已生成的文件', noInput: '暂无对应原图', noResult: '暂无对应结果',
      nextLabel: '下一步', nextDefault: '建议操作', nextDefaultDescription: '连接已有项目，即可查看图像、分析结果和复核入口。',
      goToImages: '前往图像与拼接 →', timelineTitle: '操作记录', refresh: '刷新', timelineEmpty: '操作和结果会记录在这里。',
      timelineCaption: '记录实际操作、结果和需要你处理的问题。', footer: 'MosaicAgent · 显微图像分析工作台', projectLink: '项目与说明 ↗',
      localTools: '本机工具', setupTitle: '连接现有工具', setupDescription: '填写本机路径。工作台会调用已有程序，扫描参数仍在采集工具中设置。',
      cancel: '取消', saveSettings: '保存设置', caseLabel: '图像', relatedFiles: '相关文件与说明', guide: '使用说明 ↗', repository: '代码仓库 ↗',
      openTool: '打开工具或报告 ↗', configSaved: '设置已保存。', saving: '正在保存…', requestFailed: '操作未完成', offline: '无法连接本机工作台。请确认启动窗口仍在运行，然后刷新。',
      scanMac: '扫描工具需要 Windows 仪器电脑。此电脑可查看说明、图像和分析结果。', needsSetup: '需要设置', reviewAvailable: '可查看', windowsOnly: 'Windows 采集',
      openSettings: '连接项目路径 →', noEvents: '尚无操作记录', fileUnavailable: '图像未能载入；原文件可能已移动。',
      projectRoot: '项目 / 数据文件夹', projectRootHelp: '已有项目的根目录，包含采集数据、图像或分析结果。', scanRepo: 'AmScope-Camera 文件夹',
      scanRepoHelp: '仪器电脑上已有的 AmScope-Camera 仓库路径。', inferenceResults: '层数识别结果文件夹（可选）',
      inferenceResultsHelp: '包含 results/run_manifest.json 的已有识别目录；留空读取项目默认目录。', imagesCount: '图像', logsCount: '日志', unavailable: '暂不可用',
      scanActionTitle: '从熟悉的采集工具开始', imagesActionTitle: '找到图像，检查拼接', layersActionTitle: '核对层数预测与面积统计', spectraActionTitle: '让测量与图像位置对应',
      scanActionDescription: '在仪器电脑上完成扫描，然后在工作台查看已有数据。', imagesActionDescription: '查看已有扫描图像、拼接结果和相关报告。',
      layersActionDescription: '将原图与已保存的层数预测并排核对，查看来源和统计记录。', spectraActionDescription: '查看已有光谱和位置记录，区分已测量证据与待补充数据。',
      overlay: '叠加结果', unknownTime: '已记录', close: '关闭', chooseCase: '选择图像案例', revision: '版本', requestBusy: '正在打开…',
      scanRecordsTitle: '版本与采集记录', scanRecordsDescription: '查看当前版本、操作说明与已有采集日志。',
      spectraRecordsTitle: '光谱与位置资料', spectraRecordsDescription: '查看已有测量文件与位置记录。',
      goToLayers: '前往层数分析 →', viewRecords: '查看已有资料 ↓'
    },
    en: {
      stitchSetup: 'Check new scan images & stitching settings →',
      workbench: 'ANALYSIS WORKBENCH', connecting: 'Connecting', connected: 'Local workbench', disconnected: 'Disconnected', setup: 'Setup',
      eyebrow: 'ACQUISITION TO REVIEW', headline: 'See the evidence behind every step.', subtitle: 'Choose your data. Compare the images. Decide what comes next.',
      currentRun: 'CURRENT PROJECT', noRun: 'No data selected', ready: 'Ready', selectData: 'Select data', optional: 'Optional',
      scanTitle: 'Microscope scan', scanDescription: 'Open your existing AmScope tool', imagesTitle: 'Images & stitching', imagesDescription: 'Inspect original images and mosaics',
      layersTitle: 'Layer-number analysis', layersDescription: 'U-Net layer predictions and area statistics', spectraTitle: 'Spectra & positions', spectraDescription: 'Connect measurements with evidence',
      launchScanner: 'Open AmScope scanner', scanGuide: 'Scanning guide ↗', scannerRepository: 'AmScope repository ↗',
      openImages: 'Open images & stitching results', reviewLayers: 'Review layer-number results', openSpectra: 'Open spectra & positions',
      layersHelp: 'Compare layer predictions with originals, then check counting support, valid pixels and denominators.',
      previewTitle: 'Compare before you conclude', previewSubtitle: 'Original images and analysis results, side by side.', showOverlay: 'Show overlay', original: 'Original image', result: 'Analysis result',
      openImage: 'Open full image ↗', originalEmpty: 'No original image for this step', originalEmptyHelp: 'Choose a case, or connect data in Setup',
      resultEmpty: 'No image result for this step', resultEmptyHelp: 'Only existing files appear here', noInput: 'No matching original', noResult: 'No matching result',
      nextLabel: 'NEXT STEP', nextDefault: 'Suggested action', nextDefaultDescription: 'Connect an existing project to inspect its images, results, and review tools.',
      goToImages: 'Go to images & stitching →', timelineTitle: 'Activity', refresh: 'Refresh', timelineEmpty: 'Actions and results will appear here.',
      timelineCaption: 'Actual actions, outcomes, and issues that need your attention.', footer: 'MosaicAgent · Microscopy analysis workbench', projectLink: 'Project & documentation ↗',
      localTools: 'LOCAL TOOLS', setupTitle: 'Connect your existing tools', setupDescription: 'Enter local folder paths. The workbench opens existing tools; scan parameters remain in the acquisition tool.',
      cancel: 'Cancel', saveSettings: 'Save settings', caseLabel: 'Image', relatedFiles: 'Related files & notes', guide: 'User guide ↗', repository: 'Repository ↗',
      openTool: 'Open tool or report ↗', configSaved: 'Settings saved.', saving: 'Saving…', requestFailed: 'Action could not be completed',
      offline: 'Cannot reach the local workbench. Check that its launch window is still running, then refresh.',
      scanMac: 'Scanning requires the Windows instrument computer. You can inspect guides, images, and analysis results on this computer.',
      needsSetup: 'Needs setup', reviewAvailable: 'Review available', windowsOnly: 'Windows scanner', openSettings: 'Connect a project →', noEvents: 'No activity yet',
      fileUnavailable: 'Image could not be loaded; the source file may have moved.', projectRoot: 'Project / data folder',
      projectRootHelp: 'Existing project root containing acquisition data, images, or analysis results.', scanRepo: 'AmScope-Camera folder',
      scanRepoHelp: 'Path to the existing AmScope-Camera repository on the instrument computer.', inferenceResults: 'Layer inference results folder (optional)',
      inferenceResultsHelp: 'Existing inference folder containing results/run_manifest.json; leave blank for the project default.', imagesCount: 'images', logsCount: 'logs', unavailable: 'Unavailable',
      scanActionTitle: 'Start with your familiar scanner', imagesActionTitle: 'Find your images. Inspect the mosaic.', layersActionTitle: 'Check layer predictions and area statistics',
      spectraActionTitle: 'Connect measurements to image positions', scanActionDescription: 'Acquire on the instrument computer, then inspect existing data here.',
      imagesActionDescription: 'Inspect existing scan images, stitched results, and their reports.', layersActionDescription: 'Compare originals with saved layer predictions and inspect provenance and counting records.',
      spectraActionDescription: 'Inspect existing spectra and position records, distinguishing measurements from missing evidence.',
      overlay: 'Overlay', unknownTime: 'Recorded', close: 'Close', chooseCase: 'Choose an image case', revision: 'Revision', requestBusy: 'Opening…',
      scanRecordsTitle: 'Version and acquisition records', scanRecordsDescription: 'Inspect the current version, operating guide, and existing acquisition logs.',
      spectraRecordsTitle: 'Spectral and position records', spectraRecordsDescription: 'Inspect existing measurement files and position records.',
      goToLayers: 'Go to layer-number analysis →', viewRecords: 'View existing records ↓'
    }
  };
  const ids = ['scan', 'images', 'layers', 'spectra'];
  const buttons = { scan: 'launchScannerButton', images: 'imagesButton', layers: 'layersButton', spectra: 'spectraButton' };
  const $ = id => document.getElementById(id);
  let language = 'zh';
  let activeTool = 'images';
  try {
    language = localStorage.getItem('mosaic-workbench-language') === 'en' ? 'en' : 'zh';
    const savedTool = localStorage.getItem('mosaic-workbench-tool');
    if (ids.includes(savedTool)) activeTool = savedTool;
  } catch (_) { /* Local storage may be disabled. */ }
  let state = null;
  let connected = false;
  let connectionAttempted = false;
  let pendingAction = null;
  let refreshPromise = null;
  let notice = null;
  let nextButtonAction = null;
  const selectedCases = {};
  const failedImages = new Set();
  const t = key => strings[language][key] || key;
  const localized = value => typeof value === 'string' ? value : (value && (value[language] || value.en || value.zh)) || '';

  function safeUrl(value, external = false) {
    if (typeof value !== 'string' || !value) return null;
    try {
      const url = new URL(value, location.href);
      if (!['http:', 'https:'].includes(url.protocol)) return null;
      if (!external && url.origin !== location.origin) return null;
      return url.href;
    } catch (_) { return null; }
  }

  function appendLink(parent, label, url, external = false) {
    const href = safeUrl(url, external);
    if (!href) return;
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = label;
    parent.appendChild(link);
  }

  function renderNotice() {
    const element = $('notice');
    element.replaceChildren();
    element.hidden = !notice;
    if (!notice) return;
    element.className = `notice ${notice.kind || ''}`;
    element.appendChild(document.createTextNode(localized(notice.message)));
    if (notice.url) {
      element.appendChild(document.createTextNode('  '));
      // Only links returned by the local API are used for launching tools.
      appendLink(element, t('openTool'), notice.url, true);
    }
  }

  function renderLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.title = `MosaicAgent · ${t('workbench')}`;
    document.querySelectorAll('[data-i18n]').forEach(element => { element.textContent = t(element.dataset.i18n); });
    $('languageButton').textContent = language === 'zh' ? 'English' : '中文';
    $('closeSetupButton').setAttribute('aria-label', t('close'));
    $('workflow').setAttribute('aria-label', language === 'zh' ? '分析步骤' : 'Analysis steps');
    $('caseSelect').setAttribute('aria-label', t('chooseCase'));
    render();
    renderNotice();
  }

  function renderConnection() {
    $('connection').className = `connection ${connected ? 'connected' : connectionAttempted ? 'error' : ''}`;
    $('connection').lastElementChild.textContent = t(connected ? 'connected' : connectionAttempted ? 'disconnected' : 'connecting');
  }

  function toolFor(id) { return state && Array.isArray(state.tools) ? state.tools.find(tool => tool.id === id) : null; }

  function render() {
    renderConnection();
    const project = state && state.project;
    $('runLabel').textContent = project && project.name ? project.name : t('noRun');
    $('runLabel').title = project && project.root ? project.root : '';
    document.querySelectorAll('.workflow-card').forEach(card => {
      const selected = card.dataset.stage === activeTool;
      card.classList.toggle('active', selected);
      card.setAttribute('aria-pressed', String(selected));
    });
    ids.forEach(id => {
      $(`${id}Controls`).hidden = id !== activeTool;
      const tool = toolFor(id);
      const button = $(buttons[id]);
      button.disabled = !connected || !tool || !tool.launchable || !!pendingAction;
      button.classList.toggle('busy', pendingAction === id);
      if (tool && tool.primary_label) button.textContent = localized(tool.primary_label);
      button.title = tool ? localized(tool.status_note) : '';
      const badge = $(`${id}State`);
      const statusKey = tool ? ({ ready: 'ready', needs_setup: 'needsSetup', platform_unavailable: 'windowsOnly', review_available: 'reviewAvailable' }[tool.status] || 'unavailable') : 'connecting';
      badge.textContent = t(statusKey);
      badge.className = `stage-state${tool && tool.available ? ' available' : ''}${pendingAction === id ? ' running' : ''}`;
    });
    const active = toolFor(activeTool);
    const index = ids.indexOf(activeTool) + 1;
    $('actionEyebrow').textContent = `0${index} / ${t(`${activeTool}Title`)}`;
    $('actionTitle').textContent = t(`${activeTool}ActionTitle`);
    $('actionDescription').textContent = active && active.status_note ? localized(active.status_note) : t(`${activeTool}ActionDescription`);
    const needsWindows = activeTool === 'scan' && state && state.platform !== 'Windows';
    $('platformNote').hidden = !needsWindows;
    $('platformNote').textContent = needsWindows ? t('scanMac') : '';
    renderLinks();
    renderWarnings();
    renderPreview();
    renderDetails();
    renderEvents();
    $('nextTitle').textContent = t('nextDefault');
    $('nextDescription').textContent = state && state.next_action ? localized(state.next_action) : t('nextDefaultDescription');
    const configured = !!(state && state.project && state.project.root);
    nextButtonAction = null;
    let nextButtonLabel = '';
    if (!configured) { nextButtonAction = openSetup; nextButtonLabel = 'openSettings'; }
    else if (activeTool === 'scan') { nextButtonAction = () => selectTool('images'); nextButtonLabel = 'goToImages'; }
    else if (activeTool === 'images' && toolFor('layers') && (toolFor('layers').available || toolFor('layers').launchable)) {
      nextButtonAction = () => selectTool('layers'); nextButtonLabel = 'goToLayers';
    } else if (activeTool === 'spectra' && artifactsForTool().some(artifact => ['report', 'data'].includes(artifact.role) && safeUrl(artifact.url))) {
      nextButtonAction = () => $('detailsPanel').scrollIntoView({ behavior: 'smooth', block: 'center' }); nextButtonLabel = 'viewRecords';
    }
    $('nextButton').hidden = !nextButtonAction;
    $('nextButton').textContent = nextButtonLabel ? t(nextButtonLabel) : '';
  }

  function renderWarnings() {
    const container = $('warnings');
    container.replaceChildren();
    const tool = toolFor(activeTool);
    const warnings = [...(state && Array.isArray(state.warnings) ? state.warnings : []), ...(tool && Array.isArray(tool.warnings) ? tool.warnings : [])];
    warnings.forEach(warning => {
      const message = localized(warning && warning.message || warning);
      if (!message) return;
      const paragraph = document.createElement('p');
      paragraph.textContent = message;
      container.appendChild(paragraph);
    });
    container.hidden = !container.childElementCount;
  }

  function renderLinks() {
    const scan = toolFor('scan');
    if (scan) {
      if (safeUrl(scan.guide_url, true)) $('scannerDocsLink').href = safeUrl(scan.guide_url, true);
      if (safeUrl(scan.github_url, true)) $('scannerRepoLink').href = safeUrl(scan.github_url, true);
    }
    document.querySelectorAll('[data-links]').forEach(container => {
      container.replaceChildren();
      const tool = toolFor(container.dataset.links);
      if (!tool) return;
      appendLink(container, t('guide'), tool.guide_url, true);
      appendLink(container, t('repository'), tool.github_url, true);
    });
  }

  function artifactsForTool() {
    return state && Array.isArray(state.artifacts) ? state.artifacts.filter(artifact => artifact.tool === activeTool) : [];
  }

  function caseKey(artifact) { return artifact.case_id == null ? '__default__' : String(artifact.case_id); }

  function renderPreview() {
    const images = artifactsForTool().filter(artifact => ['original', 'prediction', 'overlay'].includes(artifact.role) && safeUrl(artifact.url));
    const recordsOnly = ['scan', 'spectra'].includes(activeTool) && !images.some(artifact => ['original', 'prediction'].includes(artifact.role));
    $('previewGrid').hidden = recordsOnly;
    $('previewGrid').closest('.preview-section').classList.toggle('records-view', recordsOnly);
    $('previewTitle').textContent = t(recordsOnly ? `${activeTool}RecordsTitle` : 'previewTitle');
    const cases = [...new Set(images.map(caseKey))];
    const selected = selectedCases[activeTool];
    if (!cases.includes(selected)) selectedCases[activeTool] = cases[0] || null;
    const selector = $('caseSelect');
    selector.replaceChildren();
    cases.forEach(id => {
      const first = images.find(artifact => caseKey(artifact) === id && artifact.role === 'original') || images.find(artifact => caseKey(artifact) === id);
      const option = document.createElement('option');
      option.value = id;
      option.textContent = localized(first.case_label) || (id === '__default__' ? localized(first.label) : id);
      selector.appendChild(option);
    });
    selector.value = selectedCases[activeTool] || '';
    $('caseControl').hidden = recordsOnly || cases.length <= 1;
    const paired = images.filter(artifact => caseKey(artifact) === selectedCases[activeTool]);
    const original = paired.find(artifact => artifact.role === 'original');
    const prediction = paired.find(artifact => artifact.role === 'prediction');
    const overlay = paired.find(artifact => artifact.role === 'overlay');
    $('overlayControl').hidden = recordsOnly || !overlay;
    if (!overlay) $('overlayToggle').checked = false;
    const result = $('overlayToggle').checked && overlay ? overlay : prediction;
    renderImage('original', original);
    renderImage('result', result);
    $('resultTitle').textContent = $('overlayToggle').checked && overlay ? t('overlay') : t('result');
    const counts = state && state.counts;
    const parts = [];
    if (activeTool === 'images' && counts) {
      if (typeof counts.images === 'number') parts.push(`${counts.images} ${t('imagesCount')}`);
      if (typeof counts.logs === 'number') parts.push(`${counts.logs} ${t('logsCount')}`);
    }
    $('previewSubtitle').textContent = recordsOnly ? t(`${activeTool}RecordsDescription`) : parts.length ? parts.join(' · ') : t('previewSubtitle');
  }

  function renderImage(prefix, artifact) {
    const image = $(`${prefix}Image`);
    const empty = $(`${prefix}Empty`);
    const link = $(`${prefix}Open`);
    const meta = $(`${prefix}Meta`);
    const url = artifact && safeUrl(artifact.url);
    const failed = url && failedImages.has(url);
    image.hidden = !url || failed;
    empty.hidden = !!url && !failed;
    link.hidden = !url;
    if (url) {
      if (image.getAttribute('src') !== url) image.src = url;
      image.alt = localized(artifact.label) || t(prefix === 'original' ? 'original' : 'result');
      link.href = url;
      meta.textContent = failed ? t('fileUnavailable') : localized(artifact.label) || artifact.path || '';
      meta.title = artifact.path || '';
    } else {
      image.removeAttribute('src');
      link.removeAttribute('href');
      meta.textContent = t(prefix === 'original' ? 'noInput' : 'noResult');
      meta.title = '';
    }
  }

  function renderDetails() {
    const parent = $('detailsContent');
    parent.replaceChildren();
    const tool = toolFor(activeTool);
    const notes = tool && Array.isArray(tool.details) ? tool.details : [];
    const files = artifactsForTool().filter(artifact => ['report', 'data'].includes(artifact.role) && safeUrl(artifact.url));
    if (notes.length || (tool && tool.revision)) {
      const list = document.createElement('ul');
      notes.forEach(note => { const item = document.createElement('li'); item.textContent = localized(note); list.appendChild(item); });
      if (tool && tool.revision) { const item = document.createElement('li'); item.textContent = `${t('revision')}: ${tool.revision}`; list.appendChild(item); }
      parent.appendChild(list);
    }
    if (files.length) {
      const list = document.createElement('div');
      list.className = 'artifact-links';
      files.forEach(artifact => {
        const item = document.createElement('div');
        appendLink(item, `${localized(artifact.label) || artifact.id} ↗`, artifact.url);
        if (artifact.path) item.title = artifact.path;
        list.appendChild(item);
      });
      parent.appendChild(list);
    }
    $('detailsPanel').hidden = !parent.childElementCount;
  }

  function renderEvents() {
    const list = $('timeline');
    list.replaceChildren();
    const events = state && Array.isArray(state.events) ? state.events : [];
    if (!events.length) {
      const item = document.createElement('li');
      item.className = 'timeline-empty';
      item.textContent = t('timelineEmpty');
      list.appendChild(item);
      return;
    }
    events.slice(-12).reverse().forEach(event => {
      const item = document.createElement('li');
      item.className = ['success', 'error', 'running'].includes(event.kind) ? event.kind : '';
      const time = document.createElement('time');
      const date = new Date(event.at);
      time.textContent = Number.isNaN(date.getTime()) ? t('unknownTime') : date.toLocaleString(language === 'zh' ? 'zh-CN' : 'en-GB', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
      if (!Number.isNaN(date.getTime())) time.dateTime = date.toISOString();
      const message = document.createElement('strong');
      message.textContent = localized(event.message);
      item.append(time, message);
      list.appendChild(item);
    });
  }

  async function request(path, payload) {
    const options = payload === undefined ? { cache: 'no-store' } : {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Workbench-Token': state && state.csrf_token || '' }, body: JSON.stringify(payload)
    };
    const response = await fetch(path, options);
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`${t('requestFailed')} (HTTP ${response.status})`); }
    if (!response.ok || result.ok === false) throw new Error(localized(result.message) || localized(result.error) || `${t('requestFailed')} (HTTP ${response.status})`);
    return result;
  }

  async function refresh(manual = false) {
    if (refreshPromise) return refreshPromise;
    if (manual) {
      failedImages.clear();
      ['originalImage', 'resultImage'].forEach(id => { if ($(id).hidden) $(id).removeAttribute('src'); });
    }
    refreshPromise = (async () => {
      try {
        const result = await request('/api/state');
        if (result.app !== 'mosaic-analysis-workbench') throw new Error(t('offline'));
        state = result;
        connected = true;
        connectionAttempted = true;
        if (notice && notice.connectionError) notice = null;
        render();
        renderNotice();
      } catch (error) {
        connected = false;
        connectionAttempted = true;
        if (manual || !state) {
          notice = { message: { zh: strings.zh.offline, en: strings.en.offline }, kind: 'error', connectionError: true };
          renderNotice();
        }
        renderConnection();
        ids.forEach(id => { $(buttons[id]).disabled = true; });
      } finally { refreshPromise = null; }
    })();
    return refreshPromise;
  }

  async function performAction(id) {
    const tool = toolFor(id);
    if (!connected || !tool || !tool.launchable || pendingAction) return;
    pendingAction = id;
    notice = null;
    renderNotice();
    render();
    try {
      const result = await request('/api/action', { tool: id });
      notice = { message: result.message, kind: 'success', url: result.url };
    } catch (error) {
      notice = { message: error.message || t('requestFailed'), kind: 'error' };
    } finally {
      pendingAction = null;
      renderNotice();
      await refresh();
      render();
    }
  }

  function selectTool(id) {
    if (!ids.includes(id)) return;
    activeTool = id;
    $('overlayToggle').checked = false;
    try { localStorage.setItem('mosaic-workbench-tool', id); } catch (_) { /* Optional preference. */ }
    render();
  }

  function openSetup() {
    const container = $('setupFields');
    container.replaceChildren();
    const fields = [ ['project_root', 'projectRoot', 'projectRootHelp'], ['scan_repo', 'scanRepo', 'scanRepoHelp'], ['inference_results', 'inferenceResults', 'inferenceResultsHelp'] ];
    fields.forEach(([key, labelKey, helpKey]) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'settings-field';
      const label = document.createElement('label');
      label.htmlFor = `setting_${key}`;
      label.textContent = t(labelKey);
      const input = document.createElement('input');
      input.type = 'text'; input.id = `setting_${key}`; input.name = key; input.spellcheck = false;
      input.value = state && state.settings && state.settings[key] || '';
      const help = document.createElement('small'); help.textContent = t(helpKey);
      wrapper.append(label, input, help); container.appendChild(wrapper);
    });
    $('setupStatus').textContent = '';
    $('saveSetupButton').disabled = !connected;
    $('setupDialog').showModal();
  }

  document.querySelectorAll('[data-stage]').forEach(card => card.addEventListener('click', () => selectTool(card.dataset.stage)));
  ids.forEach(id => $(buttons[id]).addEventListener('click', () => performAction(id)));
  $('languageButton').addEventListener('click', () => {
    language = language === 'zh' ? 'en' : 'zh';
    try { localStorage.setItem('mosaic-workbench-language', language); } catch (_) { /* Optional preference. */ }
    renderLanguage();
  });
  $('caseSelect').addEventListener('change', event => { selectedCases[activeTool] = event.target.value; $('overlayToggle').checked = false; renderPreview(); });
  $('overlayToggle').addEventListener('change', renderPreview);
  $('refreshButton').addEventListener('click', () => refresh(true));
  $('setupButton').addEventListener('click', openSetup);
  $('nextButton').addEventListener('click', () => { if (nextButtonAction) nextButtonAction(); });
  ['closeSetupButton', 'cancelSetupButton'].forEach(id => $(id).addEventListener('click', () => $('setupDialog').close()));
  $('setupForm').addEventListener('submit', async event => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(event.target));
    Object.keys(payload).forEach(key => { payload[key] = payload[key].trim(); });
    $('saveSetupButton').disabled = true;
    $('setupStatus').textContent = t('saving');
    try {
      await request('/api/config', payload);
      $('setupDialog').close();
      notice = { message: { zh: strings.zh.configSaved, en: strings.en.configSaved }, kind: 'success' };
      await refresh(true);
      renderNotice();
    } catch (error) { $('setupStatus').textContent = error.message || t('requestFailed'); }
    finally { $('saveSetupButton').disabled = !connected; }
  });
  ['original', 'result'].forEach(prefix => $(`${prefix}Image`).addEventListener('error', () => {
    const src = $(`${prefix}Image`).getAttribute('src');
    if (src) failedImages.add(src);
    $(`${prefix}Image`).hidden = true;
    $(`${prefix}Empty`).hidden = false;
    $(`${prefix}Meta`).textContent = t('fileUnavailable');
  }));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  renderLanguage();
  refresh();
  setInterval(() => { if (!document.hidden && !$('setupDialog').open) refresh(); }, 5000);
})();
