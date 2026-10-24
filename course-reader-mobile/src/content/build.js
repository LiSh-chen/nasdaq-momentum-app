import { CATEGORY_LABELS, classifyMaterial } from './classify';

const YT_ID = /(?:youtube(?:-nocookie)?\.com\/(?:embed\/|watch\?(?:[^#]*&)?v=|v\/|shorts\/|live\/)|youtu\.be\/)([\w-]{11})(?![\w-])/;
const YT_LIST = /youtube(?:-nocookie)?\.com\/(?:embed\/videoseries|playlist)\?(?:[^#]*&)?list=([\w-]+)/;
const VIDEO_FILE = /\.(mp4|m4v|webm|mov)(\?|#|$)/i;
const EMBED_HOSTS = /(player\.vimeo\.com|archive\.org\/embed|bilibili\.com\/player|dailymotion\.com\/embed)/i;
const MATERIAL_EXT = /\.(pdf|pptx?|docx?|xlsx?|zip|txt|tex|ipynb|py|csv)(\?|#|$)/i;

export const extOf = (url) => (url.split(/[?#]/)[0].match(/\.([a-z0-9]+)$/i)?.[1] ?? '').toLowerCase();

const clean = (s = '') => s.replace(/\s+/g, ' ').trim();
const hostOf = (u) => { try { return new URL(u).hostname; } catch { return ''; } };

function videoFrom(src) {
  const list = src.match(YT_LIST);
  if (list) return { kind: 'youtube-playlist', ref: list[1] };
  const yt = src.match(YT_ID);
  if (yt) return { kind: 'youtube', ref: yt[1] };
  if (VIDEO_FILE.test(src)) return { kind: 'mp4', ref: src };
  if (EMBED_HOSTS.test(src)) return { kind: 'embed', ref: src };
  return null;
}

// 把頁面擷取到的原始資料，整理成分類好的課程內容
export function buildContent(raw) {
  const videos = [];
  const seen = new Set();
  const addVideo = (src, title, section) => {
    const v = videoFrom(src);
    if (!v) return;
    const key = `${v.kind}:${v.ref}`;
    if (seen.has(key)) return;
    seen.add(key);
    videos.push({ ...v, id: key, title: clean(title) || clean(section) || `影片 ${videos.length + 1}`, section: clean(section) });
  };

  for (const f of raw.frames ?? []) addVideo(f.src, f.title, f.section);
  for (const v of raw.videos ?? []) addVideo(v.src, v.title, v.section);
  for (const l of raw.links ?? []) addVideo(l.href, l.text, l.section);

  const materials = [];
  const matSeen = new Set();
  for (const l of raw.links ?? []) {
    if (!MATERIAL_EXT.test(l.href) || matSeen.has(l.href)) continue;
    matSeen.add(l.href);
    const title = clean(l.text) || decodeURIComponent(l.href.split(/[?#]/)[0].split('/').pop() || l.href);
    materials.push({
      id: l.href, title, url: l.href, ext: extOf(l.href), section: clean(l.section),
      category: classifyMaterial({ title, section: l.section, url: l.href }),
    });
  }

  // 同網站、不是檔案也不是影片的連結，視為相關頁面（可再整理）
  const here = raw.url ? raw.url.split('#')[0].replace(/\/+$/, '') : '';
  const host = hostOf(raw.url ?? '');
  const related = [];
  const relSeen = new Set();
  for (const l of raw.links ?? []) {
    const u = l.href.split('#')[0];
    const text = clean(l.text);
    if (hostOf(u) !== host || !text || text.length > 60) continue;
    if (MATERIAL_EXT.test(u) || videoFrom(u) || u.replace(/\/+$/, '') === here || relSeen.has(u)) continue;
    relSeen.add(u);
    related.push({ id: u, title: text, url: u });
    if (related.length >= 40) break;
  }

  const outline = (raw.headings ?? []).map((h) => ({ level: h.level, text: clean(h.text) })).filter((h) => h.text);
  // 太短的多半是按鈕或選單文字；中文資訊密度高，門檻訂為 8 個字以保留「授課教師：王大明」這類短資訊
const overview = [...new Set((raw.paras ?? []).map(clean).filter((s) => s.length >= 8))].slice(0, 12);

  return {
    url: raw.url,
    title: clean(raw.heading) || clean(raw.title) || raw.url,
    source: host,
    outline,
    overview,
    videos,
    materials,
    related,
  };
}

// 有內容的分頁（依固定順序）
export function tabsOf(content) {
  const by = (cat) => content.materials.filter((m) => m.category === cat);
  const tabs = [];
  if (content.overview.length || content.outline.length || by('overview').length) {
    tabs.push({ key: 'overview', label: CATEGORY_LABELS.overview, count: by('overview').length });
  }
  if (content.videos.length) tabs.push({ key: 'videos', label: CATEGORY_LABELS.videos, count: content.videos.length });
  for (const key of ['lecture', 'assignment', 'exam', 'reading', 'other']) {
    const n = by(key).length;
    if (n) tabs.push({ key, label: CATEGORY_LABELS[key], count: n });
  }
  if (content.related.length) tabs.push({ key: 'related', label: '相關頁面', count: content.related.length });
  return tabs;
}

export function playerHtml(video) {
  const frame = (src) =>
    `<iframe src="${src}" allow="autoplay; fullscreen; encrypted-media; picture-in-picture" allowfullscreen frameborder="0"></iframe>`;
  let inner;
  if (video.kind === 'youtube') {
    inner = frame(`https://www.youtube-nocookie.com/embed/${video.ref}?playsinline=1&rel=0&modestbranding=1`);
  } else if (video.kind === 'youtube-playlist') {
    inner = frame(`https://www.youtube-nocookie.com/embed/videoseries?list=${video.ref}&playsinline=1&rel=0`);
  } else if (video.kind === 'mp4') {
    inner = `<video src="${video.ref.replace(/"/g, '&quot;')}" controls playsinline preload="metadata"></video>`;
  } else {
    inner = frame(video.ref.replace(/"/g, '&quot;'));
  }
  return `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;background:#000}iframe,video{position:absolute;inset:0;width:100%;height:100%;border:0;background:#000}</style>
</head><body>${inner}</body></html>`;
}

export const youtubeWatchUrl = (video) =>
  video.kind === 'youtube' ? `https://www.youtube.com/watch?v=${video.ref}`
    : video.kind === 'youtube-playlist' ? `https://www.youtube.com/playlist?list=${video.ref}` : video.ref;

// PDF／Office 檔在手機 WebView 無法直接顯示，透過檢視服務在 App 內開啟
export const viewerUrl = (url) => `https://docs.google.com/gview?embedded=1&url=${encodeURIComponent(url)}`;
