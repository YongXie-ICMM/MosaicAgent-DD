/* Offline DOM fixture: no browser packages, network, scanner or model calls. */
'use strict';
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert/strict');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));

class Element {
  constructor(data, parent = null) {
    this.tagName = data.tag;
    this.attributes = { ...data.attrs };
    this.parentElement = parent;
    this.children = (data.children || []).map(child => new Element(child, this));
    this.ownText = data.text || '';
    this.dataset = {};
    for (const [key, value] of Object.entries(this.attributes)) {
      if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, x) => x.toUpperCase())] = value;
    }
    this.value = this.attributes.value || '';
    this.checked = Object.hasOwn(this.attributes, 'checked');
    this.hidden = Object.hasOwn(this.attributes, 'hidden');
    this.disabled = Object.hasOwn(this.attributes, 'disabled');
    this.listeners = {};
    this.className = this.attributes.class || '';
    this.classList = { toggle: (name, enabled) => {
      const names = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) names.add(name); else names.delete(name);
      this.className = [...names].join(' ');
    } };
  }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  removeAttribute(key) { delete this.attributes[key]; }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  replaceChildren(...children) { this.ownText = ''; this.children = []; this.append(...children); }
  descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
  matches(selector) {
    const attribute = selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
    if (attribute) return Object.hasOwn(this.attributes, attribute[1]) && (attribute[2] === undefined || this.attributes[attribute[1]] === attribute[2]);
    return this.tagName === selector;
  }
  querySelectorAll(selector) { return this.descendants().filter(child => child.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { return this.matches(selector) ? this : this.parentElement && this.parentElement.closest(selector); }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  fire(name, extra = {}) { for (const callback of this.listeners[name] || []) callback({ preventDefault() {}, pointerId: 1, ...extra }); }
  focus() { this.focused = true; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 600 }; }
  setPointerCapture() {}
  getContext() { return { clearRect() {}, drawImage() {}, setLineDash() {}, strokeRect() {}, fillText() {} }; }
}

async function boot(savedLanguage, blockedStorage = false) {
  const root = new Element(fixture.document);
  const all = [root, ...root.descendants()];
  const byId = Object.fromEntries(all.filter(x => x.attributes.id).map(x => [x.attributes.id, x]));
  const document = {
    documentElement: root, hidden: false,
    getElementById: id => { assert.ok(byId[id], `Missing UI element ${id}`); return byId[id]; },
    querySelectorAll: selector => all.filter(x => x.matches(selector)),
    querySelector: selector => all.find(x => x.matches(selector)) || null,
    createElement: tag => new Element({ tag, attrs: {} }), addEventListener() {}
  };
  const storage = new Map(savedLanguage === undefined ? [] : [['mosaic-workbench-language', savedLanguage]]);
  const posts = [];
  const app = { csrf_token: 'offline-test-token', platform: 'Darwin', project: { root: '/data/project', name: 'Synthetic project' } };
  const state = {
    scanner: { dd_communication_ready: true }, handover: { status: 'verified', path: '/data/scan', photos: 9 },
    prepared: [{ id: 'ticket', kind: 'stitch', status: 'prepared', bundle_dir: '/data/prepared', source_dir: '/data/scan' }],
    jobs: [{ id: 'previous', kind: 'stitch', status: 'completed', source_dir: '/data/scan', created_at: '2026-11-02T10:00:00Z', artifacts: [], output_dir: '/data/output' }, { id: 'old-spatial', kind: 'spatial', status: 'completed', source_dir: '/data/scan', created_at: '2026-11-02T10:01:00Z', artifacts: [{ name: 'SYNTHETIC_OLD_RESULT.png', url: '/old-result.png' }], output_dir: '/data/SYNTHETIC_OLD_RESULT', source_binding: { sources: { mask: { path: '/data/old_mask.npy' }, valid_support: { path: '/data/valid.npy' }, mosaic: { path: '/data/mosaic.png' } } } }],
    memory: { root: '/data/Memory', event_count: 5, workflow_recent_event_count: 2 }
  };
  const context = {
    document, console, URL, URLSearchParams,
    Image: class { constructor() { this.naturalWidth = 1000; this.naturalHeight = 600; } set src(_) { setImmediate(() => this.onload()); } },
    localStorage: { getItem: key => { if (blockedStorage) throw Error('blocked'); return storage.get(key) || null; }, setItem: (key, value) => { if (blockedStorage) throw Error('blocked'); storage.set(key, value); } },
    fetch: async (url, options) => {
      if (options.method === 'POST') {
        posts.push({ url, options });
        let result = { ok: true, message: { en: 'Synthetic job submitted.', zh: '合成任务已提交。' } };
        if (url === '/api/workflow/spatial-review') result = { review_id: 'synthetic-review', image_width: 2000, image_height: 1200, rectangles: [{ x1: 800, y1: 500, x2: 1200, y2: 700 }], status: 'needs_review', preview_urls: { optical: '/optical.png', segmentation: '/mask.png' } };
        if (url === '/api/workflow/spatial-review-kimi') result = { advisory_only: true, advice: { summary: 'Synthetic advice, not applied.', rectangles: [{ index: 0, inclusion: 'review', reason: 'Synthetic evidence', x1: 0, y1: 0, x2: 2000, y2: 1200 }] } };
        if (url === '/api/workflow/spatial') {
          const body = JSON.parse(options.body), id = 'confirmed-' + posts.length;
          const sources = Object.fromEntries([['mask', body.mask_path], ['mosaic', body.mosaic_path], ['valid_support', body.valid_path], ['exclusions', body.exclusion_path]].filter(([, file]) => file).map(([key, file]) => [key, { path: file }]));
          state.jobs.push({ id, kind: 'spatial', status: 'completed', source_dir: '/data/scan', created_at: '2026-11-02T11:00:00Z', artifacts: [{ name: 'CONFIRMED.png', url: '/confirmed.png' }], source_binding: { review_sources: sources } });
          result.job = { id };
        }
        return { ok: true, json: async () => result };
      }
      return { ok: true, json: async () => url === '/api/state' ? app : { workflow: state } };
    }
  };
  context.window = { location: { origin: 'http://localhost', search: '?step=spatial' }, setInterval: () => 1, clearInterval() {}, addEventListener() {}, open() {} };
  vm.createContext(context);
  for (const file of ['workflow_i18n.js', 'workflow.js']) vm.runInContext(fs.readFileSync(path.join(__dirname, 'static', file), 'utf8'), context, { filename: file });
  const settle = () => new Promise(resolve => setImmediate(resolve));
  await settle(); await settle();
  return { context, byId, storage, posts, settle, state };
}

(async () => {
  const { context, byId: ui, storage, posts, settle, state } = await boot();
  assert.equal(context.document.documentElement.lang, 'en');
  assert.equal(ui.languageSelect.value, 'en');
  assert.match(ui.handoverStatus.textContent, /Verified 9 images/);
  assert.match(ui.memoryCount.textContent, /5 main records/);
  assert.match(ui.startSpatial.textContent, /Review exclusion boxes/);
  assert.equal(ui.autoCentralExclusion.checked, true);
  ui.maskPath.value = '/data/native_mask.npy'; ui.validPath.value = '/data/valid.npy';
  ui.spatialMosaicPath.value = '/data/mosaic.png'; ui.sampleName.value = 'Author sample 25 mg';
  ui.maskPath.fire('input');
  assert.equal(ui.spatialProgress.hidden, true); assert.equal(ui.spatialArtifacts.children.length, 0);
  assert.match(ui.jobs.textContent, /SYNTHETIC_OLD_RESULT/);
  ui.maskPath.value = '/data/old_mask.npy'; ui.maskPath.fire('input');
  assert.equal(ui.spatialProgress.hidden, false); assert.equal(ui.spatialArtifacts.children.length, 1);
  ui.maskPath.value = '/data/native_mask.npy'; ui.maskPath.fire('input');
  assert.equal(ui.spatialProgress.hidden, true); assert.equal(ui.spatialArtifacts.children.length, 0);
  const previousSameInputs = JSON.parse(JSON.stringify(state.jobs[1]));
  previousSameInputs.id = 'same-inputs-old-regions'; previousSameInputs.source_binding.sources.mask.path = '/data/native_mask.npy';
  state.jobs.push(previousSameInputs); ui.refresh.fire('click'); await settle(); await settle();
  assert.equal(ui.spatialProgress.hidden, false);
  ui.autoCentralExclusion.checked = false;
  const stateBefore = ['maskPath', 'validPath', 'spatialMosaicPath', 'sampleName'].map(id => ui[id].value);
  ui.languageSelect.value = 'zh'; ui.languageSelect.fire('change');
  assert.equal(context.document.documentElement.lang, 'zh-CN');
  assert.match(ui.handoverStatus.textContent, /已核验 9 张照片/);
  assert.match(ui.startSpatial.textContent, /复核排除框/);
  assert.equal(storage.get('mosaic-workbench-language'), 'zh');
  assert.deepEqual(['maskPath', 'validPath', 'spatialMosaicPath', 'sampleName'].map(id => ui[id].value), stateBefore);
  assert.equal(ui.autoCentralExclusion.checked, false);
  ui.spatialForm.fire('submit'); await settle(); await settle(); await settle();
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/api/workflow/spatial-review');
  assert.equal(ui.confirmSpatial.disabled, true);
  assert.equal(ui.spatialReview.hidden, false);
  assert.equal(ui.spatialProgress.hidden, true); // Old regions must not appear verified for this new review.
  // Add and adjust a rectangle in native pixels (preview is downsampled by 2).
  ui.spatialMaskCanvas.fire('pointerdown', { clientX: 100, clientY: 200 });
  ui.spatialMaskCanvas.fire('pointermove', { clientX: 150, clientY: 250 });
  ui.spatialMaskCanvas.fire('pointerup');
  assert.equal(ui.rectX1.value, '200'); assert.equal(ui.rectY1.value, '400');
  ui.rectX2.value = '350'; ui.rectX2.fire('input'); ui.applyRectangle.fire('click');
  ui.reviewKimi.fire('click'); await settle(); await settle();
  assert.match(ui.reviewKimiAdvice.textContent, /Synthetic advice/);
  assert.equal(ui.confirmSpatial.disabled, true);
  assert.equal(posts.filter(post => post.url === '/api/workflow/spatial').length, 0);
  ui.confirmReview.checked = true; ui.confirmReview.fire('change');
  assert.equal(ui.confirmSpatial.disabled, false);
  ui.confirmSpatial.fire('click'); await settle(); await settle();
  const finals = posts.filter(post => post.url === '/api/workflow/spatial');
  assert.equal(finals.length, 1);
  const submitted = JSON.parse(finals[0].options.body);
  assert.equal(submitted.review_confirmed, true); assert.equal(submitted.review_id, 'synthetic-review');
  assert.equal(submitted.mosaic_path, '/data/mosaic.png'); assert.equal(submitted.auto_central_exclusion, false);
  assert.deepEqual(submitted.rectangles, [{ x1: 800, y1: 500, x2: 1200, y2: 700 }, { x1: 200, y1: 400, x2: 350, y2: 500 }]);
  assert.equal(submitted.mask_path, '/data/native_mask.npy');
  assert.equal(ui.spatialProgress.hidden, false);
  assert.equal(posts[0].options.headers['X-Workbench-Token'], 'offline-test-token');
  ui.languageSelect.value = 'en'; ui.languageSelect.fire('change');
  assert.match(ui.startSpatial.textContent, /Review exclusion boxes/);
  ui.clearRectangles.fire('click'); assert.equal(ui.reviewKimiAdvice.hidden, true);
  assert.equal(ui.confirmSpatial.disabled, true);
  assert.equal(ui.spatialProgress.hidden, true); // Clearing boxes invalidates the current result, too.
  ui.confirmReview.checked = true; ui.confirmReview.fire('change'); ui.confirmSpatial.fire('click'); await settle(); await settle();
  const cleared = posts.filter(post => post.url === '/api/workflow/spatial').at(-1);
  assert.deepEqual(JSON.parse(cleared.options.body).rectangles, []);
  ui.spatialMosaicPath.value = ''; ui.spatialMosaicPath.fire('input'); ui.autoCentralExclusion.checked = true;
  assert.equal(ui.spatialReview.hidden, true); assert.equal(ui.confirmSpatial.disabled, true);
  const before = posts.length; ui.confirmSpatial.fire('click'); await settle(); assert.equal(posts.length, before);
  ui.spatialForm.fire('submit'); await settle(); await settle(); await settle();
  assert.equal(posts.length, before); assert.match(ui.notice.textContent, /matching optical mosaic/);
  const chinese = await boot('zh'); assert.equal(chinese.context.document.documentElement.lang, 'zh-CN');
  const invalid = await boot('invalid'); assert.equal(invalid.context.document.documentElement.lang, 'en');
  const blocked = await boot(undefined, true); assert.equal(blocked.context.document.documentElement.lang, 'en');
  const i18n = context.window.DDWorkflowI18n;
  assert.equal(i18n.localized({ en: 'English evidence', zh: '中文证据' }), 'English evidence');
  assert.equal(i18n.localized('/data/原始照片/sample.png'), '/data/原始照片/sample.png');
  for (const key of fixture.translationKeys) assert.ok(Object.hasOwn(i18n.english, key), `Missing translation: ${key}`);
  console.log('Offline UI checks passed: English/Chinese preferences, native box edits, advisory-only Kimi, explicit human confirmation and source-change invalidation.');
})().catch(error => { console.error(error); process.exitCode = 1; });
