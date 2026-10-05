(() => {
  'use strict';

  const strings = {
    zh: {
      skip: '跳到配置', setupTitle: '拼接配置', back: '返回分析工作台', eyebrow: 'RAW TILES → STITCHING PROFILE',
      headline: '先确认图块，再准备拼接。', subtitle: '按顺序检查文件夹、确认采集几何，保存一份可以复核的配置与命令。',
      scope: '这里只检查本机文件并保存配置。保存后，在终端中运行命令才会开始拼接。',
      inspectTitle: '检查原始图块', inspectIntro: '填写运行工作台的电脑上的文件夹路径。选择原始图块，而不是已经拼好的大图。',
      dataDir: '原始图块文件夹', dataDirHelp: '在文件管理器中复制完整路径并粘贴到这里。此操作不会上传图像。',
      layoutPath: '已有布局 JSON 文件路径（可选）', layoutHelp: '只有已知布局文件时才填写；留空会尝试识别受支持的图块命名和列文件夹结构。',
      inspectButton: '检查文件夹', inspecting: '正在检查…', inspected: '已检查', needsInspect: '等待检查',
      acquisitionTitle: '已找到采集记录', acquisitionSize: '采集预期 / 原图实测', acquisitionGrid: '采集网格 / 当前选择', acquisitionSteps: '记录的平台位移', acquisitionStatus: '记录状态 / 选择范围',
      calibrationPending: '像素位移标定仍待核实。当前尺寸和位移台步数不会自动换算成旧 4K 拼接参数。', unrecorded: '未记录', subset: '部分图块', partial_or_subset: '未完成采集或部分图块', full_recorded_grid: '完整记录网格', completed: '记录为完成', running: '仍在采集', unknownStatus: '状态未知',
      tileCount: '图块数量', imageSize: '单张像素尺寸', fileSize: '单张文件大小', layoutMode: '布局来源',
      sizeNote: 'MB 受像素尺寸、编码、压缩和内容共同影响，不能单独作为分辨率或几何标定依据。像素尺寸也不能单独证明视野、物镜或位移相同。',
      profileTitle: '明确选择几何配置', profileIntro: '根据采集记录选择；无法确认的参数，请先向采集者核实。', profileChoice: '选择一个配置',
      historicalTitle: '历史配置 · 3840 × 2160', historicalHelp: '仅用于原始历史采集几何：3840 × 2160 像素，沿用 Figure 3 的标定与平台步数。相同尺寸本身不足以确认适用。',
      historicalUnavailable: '当前图块不是 3840 × 2160，此选项不可用。',
      rescaledTitle: '同一视野的等比例缩放', rescaledHelp: '仅用于历史图块整体等比例缩放。像素位移随尺寸缩放，物理平台步数不变；裁剪、换物镜或改变视野不适用。',
      customTitle: '自定义 · 已标定的像素位移', customHelp: '填写当前原始像素坐标下的两个相邻图块位移向量，并记录标定来源。',
      confirmTitle: '对照采集记录，逐项确认', sameOptics: '相机与物镜等光学配置与历史采集相同。',
      sameFieldOfView: '物理视野相同，整幅图像只做了等比例缩放，没有裁剪。', sameStageSteps: '平台实际执行的位移步数与历史采集相同。',
      vectorsTitle: '原始像素位移（包含符号）', vectorsHelp: 'v 表示同一列内相邻图块，h 表示相邻列；dx、dy 是该位移在图像 x、y 方向的分量。单位为当前输入图块的像素，不是平台步数。',
      dxV: '列内位移 dx_v (px)', dyV: '列内位移 dy_v (px)', dxH: '跨列位移 dx_h (px)', dyH: '跨列位移 dy_h (px)',
      calibrationNote: '标定依据', calibrationHelp: '记录相机、物镜、视野、平台位移及标定日期或文件。不要根据文件 MB 猜测向量。',
      resolutionTitle: '缓存与输出分辨率', scaleDiv: '缓存缩小倍数 (scale_div)', scaleDivHelp: '例如 8 表示宽、高各缩小到 1/8。选项已按实际图块尺寸筛选。',
      outScale: '输出比例 (out_scale)', outScaleHelp: '相对于原始图块：0.125 = 1/8，1 = 原始像素尺度。', full: '渲染时读取原始图块（full）',
      fullHelp: '勾选后可使用更高输出比例，读取时间与内存需求可能增加；输出仍由 out_scale 决定。未勾选时，输出比例不能高于 1 / scale_div。',
      saveTitle: '保存配置与命令', saveIntro: '保存到单独的配置文件夹，之后可检查参数、保留记录并在终端运行。', outputParent: '配置保存位置（父文件夹）',
      outputHelp: '检查后会填入建议路径。每次保存会创建独立的配置包，保留图块检查记录和命令参数。', prepareButton: '保存配置并生成命令', preparing: '正在保存…',
      inspectFirst: '先完成第 1 步检查。', chooseProfile: '请选择一个几何配置。', confirmRequired: '请完成同视野缩放的三项确认。', customRequired: '请填写四个位移分量与标定依据。',
      scaleRequired: '请填写有效的缓存倍数和输出比例（大于 0，且不超过 1）。', resolutionRequired: '此输出比例高于缓存分辨率，请降低比例或勾选读取原始图块。',
      outputRequired: '请填写配置保存位置。', ready: '可以保存配置', profileSelected: '已选择',
      preparedLabel: 'PROFILE PREPARED', preparedTitle: '配置已保存，尚未执行拼接。',
      preparedHelp: '先查看配置包中的说明和检查记录，再在已安装 MosaicAgent 依赖的终端中运行命令。生成配置不代表图块配准或拼接已通过验证。',
      commandTitle: '拼接命令', copy: '复制命令', copied: '命令已复制。', copyFallback: '自动复制不可用。命令已选中，请按 Ctrl+C 或 ⌘C 复制。',
      commandHelp: '在配置包的说明文件中查看运行步骤。请使用上方注明的终端：Windows 使用 PowerShell；macOS / Linux 使用 POSIX shell。argv JSON 保存精确参数。',
      footer: 'MosaicAgent · 本机拼接配置', connecting: '正在连接本机工作台…', connected: '本机工作台已连接', disconnected: '未连接；检查时会重试',
      offline: '无法连接本机工作台。请确认启动窗口仍在运行，再重试。', failed: '操作未完成，请检查输入后重试。',
      invalidInspection: '检查接口未返回完整的图块信息，请更新或重启本机工作台后重试。', invalidPrepared: '配置接口未返回完整的命令信息，请检查本机工作台日志。',
      inspectDone: '文件检查完成。请根据实际采集记录选择配置。', pathsChanged: '路径已改变，请重新检查文件夹。', settingsChanged: '参数已改变，请重新保存配置并生成命令。原先保存的配置包仍保留在磁盘上。',
      'flat-grid': '图块文件名网格', 'named-columns': '命名的列文件夹', 'provided-layout': '指定的布局 JSON',
      bundle_dir: '配置包文件夹', profile_path: '几何配置 JSON', preflight_path: '检查记录', argv_path: '精确命令参数 JSON', layer_input_contract_path: '层数分析输入记录', instructions_path: '运行说明', shellLabel: '命令格式'
    },
    en: {
      skip: 'Skip to setup', setupTitle: 'STITCH SETUP', back: 'Back to workbench', eyebrow: 'RAW TILES → STITCHING PROFILE',
      headline: 'Check your tiles. Prepare your stitch.', subtitle: 'Inspect the folder, confirm the acquisition geometry, and save a profile and command for review.',
      scope: 'This page checks local files and saves configuration. Stitching starts only when you run the saved command in a terminal.',
      inspectTitle: 'Inspect the raw tiles', inspectIntro: 'Enter a folder on the computer running the workbench. Choose raw tiles, rather than an existing mosaic.',
      dataDir: 'Raw tile folder', dataDirHelp: 'Copy the full path from your file manager and paste it here. Images stay on this computer.',
      layoutPath: 'Existing layout JSON path (optional)', layoutHelp: 'Fill this only if you have a known layout file. Otherwise the checker tries supported tile names and column folders.',
      inspectButton: 'Inspect folder', inspecting: 'Inspecting…', inspected: 'Inspected', needsInspect: 'Awaiting inspection',
      acquisitionTitle: 'Acquisition record found', acquisitionSize: 'Expected / measured input size', acquisitionGrid: 'Recorded grid / selected tiles', acquisitionSteps: 'Recorded stage displacement', acquisitionStatus: 'Recorded status / selection scope',
      calibrationPending: 'Pixel displacement calibration remains unverified. Input dimensions and recorded stage steps do not automatically determine the historical 4K stitching parameters.', unrecorded: 'Not recorded', subset: 'Subset', partial_or_subset: 'Partial acquisition or subset', full_recorded_grid: 'Full recorded grid', completed: 'Recorded as completed', running: 'Still acquiring', unknownStatus: 'Unknown status',
      tileCount: 'Tiles', imageSize: 'Pixels per tile', fileSize: 'File size per tile', layoutMode: 'Layout source',
      sizeNote: 'File size depends on dimensions, encoding, compression and content; it cannot by itself determine resolution or acquisition geometry. Pixel dimensions alone also cannot establish the field of view, objective or stage movement.',
      profileTitle: 'Choose the geometry explicitly', profileIntro: 'Use the acquisition record. Check unknown parameters with the person who acquired the images.', profileChoice: 'Choose one profile',
      historicalTitle: 'Historical · 3840 × 2160', historicalHelp: 'For the original historical acquisition geometry only: 3840 × 2160 pixels, with the Figure 3 calibration and stage steps. Matching dimensions alone do not establish compatibility.',
      historicalUnavailable: 'Current tiles are not 3840 × 2160, so this option is unavailable.',
      rescaledTitle: 'Proportional resize of the same field of view', rescaledHelp: 'Only for uniformly resized historical tiles. Pixel displacements scale with dimensions; physical stage steps stay unchanged. Crops, different objectives or a different field of view do not qualify.',
      customTitle: 'Custom · calibrated pixel displacements', customHelp: 'Enter two neighboring-tile displacement vectors in current input pixels, and record the calibration source.',
      confirmTitle: 'Confirm each item against the acquisition record', sameOptics: 'The camera, objective and other optical settings match the historical acquisition.',
      sameFieldOfView: 'The physical field of view is unchanged; the entire image was uniformly resized without cropping.', sameStageSteps: 'The actual stage displacement steps match the historical acquisition.',
      vectorsTitle: 'Displacements in input pixels (including signs)', vectorsHelp: 'v is between neighboring tiles in one column; h is between neighboring columns. dx and dy are image x and y components. Use current input pixels, not stage steps.',
      dxV: 'Within column dx_v (px)', dyV: 'Within column dy_v (px)', dxH: 'Across columns dx_h (px)', dyH: 'Across columns dy_h (px)',
      calibrationNote: 'Calibration evidence', calibrationHelp: 'Record the camera, objective, field of view, stage movement, and calibration date or file. Do not infer vectors from file size in MB.',
      resolutionTitle: 'Cache and output resolution', scaleDiv: 'Cache divisor (scale_div)', scaleDivHelp: 'For example, 8 divides width and height by 8. Choices are filtered for the actual tile dimensions.',
      outScale: 'Output scale (out_scale)', outScaleHelp: 'Relative to input tiles: 0.125 = 1/8; 1 = original pixel scale.', full: 'Read original tiles while rendering (full)',
      fullHelp: 'This allows a higher output scale and may increase reading time and memory use. out_scale still determines output resolution. When unchecked, output scale cannot exceed 1 / scale_div.',
      saveTitle: 'Save the profile and command', saveIntro: 'Create a separate configuration bundle to review parameters, retain records and run from a terminal.', outputParent: 'Save location (parent folder)',
      outputHelp: 'Inspection fills in a suggested path. Each save creates a separate bundle containing the tile inspection record and command arguments.', prepareButton: 'Save profile & generate command', preparing: 'Saving…',
      inspectFirst: 'Complete the folder inspection in step 1.', chooseProfile: 'Choose a geometry profile.', confirmRequired: 'Complete all three same-field-of-view confirmations.', customRequired: 'Enter four displacement components and calibration evidence.',
      scaleRequired: 'Enter a valid cache divisor and output scale greater than 0 and at most 1.', resolutionRequired: 'Output scale exceeds cache resolution. Reduce it or enable reading original tiles.',
      outputRequired: 'Enter a save location.', ready: 'Ready to save profile', profileSelected: 'Selected',
      preparedLabel: 'PROFILE PREPARED', preparedTitle: 'Profile saved. Stitching has not run.',
      preparedHelp: 'Review the instructions and inspection record in the bundle, then run the command in a terminal with MosaicAgent dependencies installed. Preparing a profile does not validate tile registration or a stitched result.',
      commandTitle: 'Stitching command', copy: 'Copy command', copied: 'Command copied.', copyFallback: 'Automatic copying is unavailable. The command is selected; press Ctrl+C or ⌘C to copy.',
      commandHelp: 'See the bundle instructions for the next steps. Use the terminal format shown above: PowerShell on Windows, or a POSIX shell on macOS / Linux. The argv JSON preserves exact arguments.',
      footer: 'MosaicAgent · Local stitch setup', connecting: 'Connecting to local workbench…', connected: 'Local workbench connected', disconnected: 'Disconnected; inspection will retry',
      offline: 'Cannot reach the local workbench. Check that its launch window is still running and retry.', failed: 'The action could not be completed. Check your input and retry.',
      invalidInspection: 'The inspection API did not return complete tile information. Update or restart the local workbench and retry.', invalidPrepared: 'The profile API did not return a complete command. Check the local workbench log.',
      inspectDone: 'Files inspected. Choose a profile using the actual acquisition record.', pathsChanged: 'Paths changed. Inspect the folder again.', settingsChanged: 'Settings changed. Save again to generate an updated command. The previous bundle remains on disk.',
      'flat-grid': 'Grid tile names', 'named-columns': 'Named column folders', 'provided-layout': 'Provided layout JSON',
      bundle_dir: 'Bundle folder', profile_path: 'Geometry profile JSON', preflight_path: 'Inspection record', argv_path: 'Exact argument JSON', layer_input_contract_path: 'Layer-analysis input record', instructions_path: 'Run instructions', shellLabel: 'Command format'
    }
  };
  const $ = id => document.getElementById(id);
  const t = key => strings[language][key] || key;
  const vectorIds = ['dxV', 'dyV', 'dxH', 'dyH'];
  const confirmationIds = ['sameOptics', 'sameFieldOfView', 'sameStageSteps'];
  let language = 'zh';
  let csrfToken = null;
  let connection = 'connecting';
  let inspection = null;
  let inspectedPaths = null;
  let prepared = null;
  let notice = null;
  let copyStatus = null;
  let busy = null;
  let revision = 0;

  const localize = value => typeof value === 'string' ? value : (value && (value[language] || value.en || value.zh)) || t('failed');
  const paths = () => ({ data_dir: $('dataDir').value.trim(), layout_path: $('layoutPath').value.trim() });
  const profileKind = () => document.querySelector('input[name="profile_kind"]:checked')?.value || '';
  const samePaths = (a, b) => a && b && a.data_dir === b.data_dir && a.layout_path === b.layout_path;

  function setNotice(key, kind = 'info', message = null) {
    notice = key || message ? { key, kind, message } : null;
    renderNotice();
  }

  function renderNotice() {
    $('notice').hidden = !notice;
    $('notice').className = `notice ${notice?.kind || ''}`;
    $('notice').textContent = notice ? (notice.key ? t(notice.key) : localize(notice.message)) : '';
  }

  function showWarnings(id, warnings) {
    const container = $(id);
    container.replaceChildren();
    const items = Array.isArray(warnings) ? warnings : [];
    container.hidden = items.length === 0;
    if (!items.length) return;
    const list = document.createElement('ul');
    items.forEach(warning => {
      const item = document.createElement('li');
      item.textContent = localize(warning);
      list.appendChild(item);
    });
    container.appendChild(list);
  }

  function readiness() {
    if (!inspection || !samePaths(paths(), inspectedPaths)) return 'inspectFirst';
    const kind = profileKind();
    if (!kind) return 'chooseProfile';
    if (kind === 'rescaled' && confirmationIds.some(id => !$(id).checked)) return 'confirmRequired';
    if (kind === 'custom' && (vectorIds.some(id => !$(id).value.trim() || !Number.isFinite(Number($(id).value))) || !$('calibrationNote').value.trim())) return 'customRequired';
    const divisor = Number($('scaleDiv').value);
    const scale = Number($('outScale').value);
    if (!Number.isInteger(divisor) || divisor < 1 || !Number.isFinite(scale) || scale <= 0 || scale > 1) return 'scaleRequired';
    if (!$('full').checked && scale > 1 / divisor + 1e-12) return 'resolutionRequired';
    if (!$('outputParent').value.trim()) return 'outputRequired';
    return 'ready';
  }

  function renderControls() {
    const kind = profileKind();
    $('inspectButton').disabled = !!busy;
    $('inspectButton').textContent = t(busy === 'inspect' ? 'inspecting' : 'inspectButton');
    $('inspectForm').setAttribute('aria-busy', String(busy === 'inspect'));
    $('prepareForm').setAttribute('aria-busy', String(busy === 'prepare'));
    $('profileFields').disabled = !inspection || !!busy;
    $('saveFields').disabled = !inspection || !!busy;
    $('profileState').textContent = inspection ? t(kind ? 'profileSelected' : 'chooseProfile') : '';
    $('inspectState').textContent = t(busy === 'inspect' ? 'inspecting' : inspection ? 'inspected' : 'needsInspect');
    $('connectionState').textContent = t(connection);
    $('rescaledFields').hidden = kind !== 'rescaled';
    $('customFields').hidden = kind !== 'custom';
    confirmationIds.forEach(id => { $(id).required = kind === 'rescaled'; });
    [...vectorIds, 'calibrationNote'].forEach(id => { $(id).required = kind === 'custom'; });
    const historical = document.querySelector('input[value="historical"]');
    historical.disabled = !inspection || inspection.image_size[0] !== 3840 || inspection.image_size[1] !== 2160;
    $('historicalAvailability').textContent = inspection && historical.disabled ? t('historicalUnavailable') : '';
    $('prepareButton').disabled = !!busy || readiness() !== 'ready';
    $('prepareButton').textContent = t(busy === 'prepare' ? 'preparing' : 'prepareButton');
    $('prepareHint').textContent = t(readiness());
  }

  function renderInspection() {
    $('inspectionResult').hidden = !inspection;
    if (!inspection) return;
    $('tileCount').textContent = String(inspection.tile_count);
    $('imageSize').textContent = `${inspection.image_size[0]} × ${inspection.image_size[1]} px`;
    const mb = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString(language === 'zh' ? 'zh-CN' : 'en', { maximumFractionDigits: 2 }) : '—';
    $('fileSize').textContent = `${mb(inspection.file_mb_min)}–${mb(inspection.file_mb_max)} MB`;
    $('layoutMode').textContent = t(inspection.layout_mode);
    const acquisition = inspection.acquisition;
    $('acquisitionRecord').hidden = !acquisition?.present;
    if (acquisition?.present) {
      const sizeText = value => Array.isArray(value) ? `${value[0]} × ${value[1]} px` : t('unrecorded');
      const grid = acquisition.grid || {};
      const steps = acquisition.steps || {};
      $('acquisitionSize').textContent = `${sizeText(acquisition.expected_image_size)} / ${sizeText(acquisition.actual_image_size)}`;
      $('acquisitionGrid').textContent = `${grid.nx} × ${grid.ny} / ${grid.selected_tiles} / ${grid.planned_tiles}`;
      $('acquisitionSteps').textContent = `dx: ${steps.dx_steps ?? t('unrecorded')}; dy: ${steps.dy_steps ?? t('unrecorded')} (stage steps)`;
      $('acquisitionStatus').textContent = `${t(acquisition.session_status || 'unknownStatus')} / ${t(grid.selection_scope || 'unknownStatus')}`;
    }
    showWarnings('inspectWarnings', inspection.warnings);
  }

  function renderPrepared() {
    $('preparedResult').hidden = !prepared;
    $('copyStatus').textContent = copyStatus ? t(copyStatus) : '';
    if (!prepared) return;
    $('savedPaths').replaceChildren();
    ['bundle_dir', 'profile_path', 'preflight_path', 'layer_input_contract_path', 'argv_path', 'instructions_path'].forEach(key => {
      if (!prepared[key]) return;
      const row = document.createElement('div');
      const term = document.createElement('dt');
      const description = document.createElement('dd');
      term.textContent = t(key);
      description.textContent = String(prepared[key]);
      row.append(term, description);
      $('savedPaths').appendChild(row);
    });
    $('command').textContent = prepared.command;
    $('command').setAttribute('aria-label', t('commandTitle'));
    $('commandShell').textContent = `${t('shellLabel')}: ${prepared.command_shell || '—'}`;
    showWarnings('prepareWarnings', prepared.warnings);
  }

  function renderLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.title = `MosaicAgent · ${t('setupTitle')}`;
    document.querySelectorAll('[data-i18n]').forEach(element => { element.textContent = t(element.dataset.i18n); });
    $('languageButton').textContent = language === 'zh' ? 'English' : '中文';
    renderNotice();
    renderInspection();
    renderPrepared();
    renderControls();
  }

  async function connect() {
    connection = 'connecting';
    renderControls();
    try {
      const response = await fetch('/api/state', { credentials: 'same-origin', cache: 'no-store' });
      if (!response.ok) throw new Error();
      const state = await response.json();
      if (typeof state.csrf_token !== 'string' || !state.csrf_token) throw new Error();
      csrfToken = state.csrf_token;
      connection = 'connected';
    } catch (_) {
      csrfToken = null;
      connection = 'disconnected';
    }
    renderControls();
    return !!csrfToken;
  }

  async function post(endpoint, payload) {
    if (!csrfToken && !await connect()) throw { key: 'offline' };
    let response;
    try {
      response = await fetch(endpoint, {
        method: 'POST', credentials: 'same-origin', cache: 'no-store',
        headers: { 'Content-Type': 'application/json', 'X-Workbench-Token': csrfToken },
        body: JSON.stringify(payload)
      });
    } catch (_) {
      connection = 'disconnected';
      csrfToken = null;
      throw { key: 'offline' };
    }
    let data;
    try { data = await response.json(); } catch (_) { throw { key: 'failed' }; }
    if (!response.ok || data.ok !== true) {
      if (response.status === 403) csrfToken = null;
      throw { message: data.message || data.error || null, key: data.message || data.error ? null : 'failed' };
    }
    connection = 'connected';
    return data;
  }

  function invalidatePrepared() {
    revision += 1;
    if (prepared) setNotice('settingsChanged');
    prepared = null;
    copyStatus = null;
    $('command').textContent = '';
    $('savedPaths').replaceChildren();
    renderPrepared();
  }

  function invalidatePaths() {
    const hadInspection = !!inspection;
    invalidatePrepared();
    inspection = null;
    inspectedPaths = null;
    document.querySelectorAll('input[name="profile_kind"]').forEach(input => { input.checked = false; });
    confirmationIds.forEach(id => { $(id).checked = false; });
    if (hadInspection || notice?.key === 'inspectDone') setNotice('pathsChanged');
    renderInspection();
    renderControls();
  }

  $('inspectForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !$('inspectForm').reportValidity()) return;
    invalidatePaths();
    const submittedPaths = paths();
    const requestRevision = revision;
    busy = 'inspect';
    setNotice(null);
    renderControls();
    try {
      const result = await post('/api/stitch/inspect', submittedPaths);
      if (revision !== requestRevision || !samePaths(paths(), submittedPaths)) return;
      const info = result.inspection;
      if (result.status !== 'inspected' || !info || !info.fingerprint || !Array.isArray(info.image_size) || info.image_size.length !== 2 || !Array.isArray(info.scale_div_choices) || !info.scale_div_choices.length) throw { key: 'invalidInspection' };
      inspection = { ...info, warnings: result.warnings };
      inspectedPaths = submittedPaths;
      $('scaleDiv').replaceChildren();
      info.scale_div_choices.forEach(divisor => {
        const option = document.createElement('option');
        option.value = String(divisor);
        option.textContent = `${divisor} (1/${divisor})`;
        $('scaleDiv').appendChild(option);
      });
      $('scaleDiv').value = String(info.default_scale_div);
      $('outScale').value = String(1 / Number($('scaleDiv').value));
      $('outputParent').value = info.default_output_parent || '';
      $('full').checked = false;
      setNotice('inspectDone');
      renderInspection();
    } catch (error) {
      if (revision === requestRevision) setNotice(error.key || null, 'error', error.message || (error.key ? null : t('failed')));
    } finally {
      busy = null;
      renderControls();
    }
  });

  $('prepareForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy) return;
    const reason = readiness();
    if (reason !== 'ready') { setNotice(reason, 'error'); return; }
    if (!$('prepareForm').reportValidity()) return;
    invalidatePrepared();
    const requestRevision = revision;
    const payload = {
      ...inspectedPaths, fingerprint: inspection.fingerprint, profile_kind: profileKind(),
      confirmations: { same_optics: $('sameOptics').checked, same_field_of_view: $('sameFieldOfView').checked, same_stage_steps: $('sameStageSteps').checked },
      scale_div: Number($('scaleDiv').value), out_scale: Number($('outScale').value), full: $('full').checked,
      output_parent: $('outputParent').value.trim()
    };
    if (payload.profile_kind === 'custom') {
      payload.nominal_vectors = vectorIds.map(id => Number($(id).value));
      payload.calibration_note = $('calibrationNote').value.trim();
    }
    busy = 'prepare';
    setNotice(null);
    renderControls();
    try {
      const result = await post('/api/stitch/prepare', payload);
      if (revision !== requestRevision) return;
      if (result.status !== 'profile_prepared' || typeof result.command !== 'string' || !result.command || !result.bundle_dir) throw { key: 'invalidPrepared' };
      prepared = result;
      renderPrepared();
      $('preparedResult').focus();
    } catch (error) {
      if (revision === requestRevision) setNotice(error.key || null, 'error', error.message || (error.key ? null : t('failed')));
    } finally {
      busy = null;
      renderControls();
    }
  });

  ['dataDir', 'layoutPath'].forEach(id => $(id).addEventListener('input', invalidatePaths));
  $('prepareForm').addEventListener('input', () => { invalidatePrepared(); renderControls(); });
  $('languageButton').addEventListener('click', () => { language = language === 'zh' ? 'en' : 'zh'; renderLanguage(); });
  $('copyButton').addEventListener('click', async () => {
    if (!prepared) return;
    const command = prepared.command;
    try {
      if (!navigator.clipboard?.writeText) throw new Error();
      await navigator.clipboard.writeText(command);
      if (prepared?.command === command) copyStatus = 'copied';
    } catch (_) {
      if (prepared?.command !== command) return;
      const range = document.createRange();
      range.selectNodeContents($('command'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      $('command').focus();
      copyStatus = 'copyFallback';
    }
    $('copyStatus').textContent = copyStatus ? t(copyStatus) : '';
  });

  renderLanguage();
  connect();
})();
