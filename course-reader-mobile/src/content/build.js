import { CATEGORY_LABELS, classifyMaterial } from './classify';

const YT_ID = /(?:youtube(?:-nocookie)?\.com\/(?:embed\/|watch\?(?:[^#]*&)?v=|v\/|shorts\/|live\/)|youtu\.be\/)([\w-]{11})(?![\w-])/;
const YT_LIST = /youtube(?:-nocookie)?\.com\/(?:embed\/videoseries|playlist)\?(?:[^#]*&)?list=([\w-]+)/;
const VIDEO_FILE = /\.(mp4|m4v|webm|mov)(\?|#|$)/i;
const EMBED_HOSTS = /(player\.vimeo\.com|archive\.org\/embed|bilibili\.com\/player|dailymotion\.com\/embed)/i;
const MATERIAL_EXT = /\.(pdf|pptx?|docx?|xlsx?|zip|txt|tex|ipynb|py|csv)(\?|#|$)/i;

export const extOf = (url) => (url.split(/[?#]/)[0].match(/\.([a-z0-9]+)$/i)?.[1] ?? '').toLowerCase();

const clean = (s = '') => s.replace(/\s+/g, ' ').trim();

// OCW 的講義、作業連到「資源頁」（/resources/<名稱>/），網址不一定帶副檔名，要靠連結文字與所在區塊判斷
const RESOURCE_PAGE = /\/resources\/[^/?#]+\/?(?:[?#]|$)/i;
const EXT_HINT = /\b(pdf|pptx?|docx?|xlsx?|zip)\b/i;
const EXT_HINT_PAREN = /\s*[(\[（]\s*(pdf|pptx?|docx?|xlsx?|zip)\s*[)\]）]/i;
const PAGER = /^[«»‹›<>|\s]*(previous|next|prev|上一[頁章篇]?|下一[頁章篇]?)[«»‹›<>|\s]*$/i;
const VIDEO_WORDS = /video|影片|mp4/i;
// 連結文字沒有意義時（語言名稱、download、檔案大小…），改用檔名當標題
const GENERIC_TEXT = /^(english|chinese|中文|繁體中文|简体中文|download( the)?( video| file| pdf)?|下載|view|pdf|video( \d+(\.\d+)? ?[kmg]b)?|here|link|more|read more|click here)$/i;

export function titleFromUrl(url) {
  let name = '';
  try { name = decodeURIComponent(new URL(url).pathname.split('/').filter(Boolean).pop() || ''); } catch { name = url; }
  return name
    .replace(/\.[a-z0-9]{2,4}$/i, '')
    .replace(/^[0-9a-f]{16,}[_-]?/i, '') // OCW 檔名常是「雜湊_檔名」，去掉前面的雜湊
    .replace(/_(pdf|pptx?|docx?|xlsx?|zip|mp4)$/i, '')
    .replace(/[_-]?(360|480|720|1080)p$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim();
}
// 檔名是一長串雜湊（如 637ca1c7a1cac1f4…）時沒有參考價值，改用連結文字或所在區塊的標題
const HASHY = /^[0-9a-f]{16,}$/i;
const isHashy = (s) => HASHY.test(s.replace(/[\s_-]+/g, ''));

function titleOf(text, url, section = '') {
  const t = clean(text);
  if (t && !GENERIC_TEXT.test(t)) return t;
  const fromUrl = titleFromUrl(url);
  if (fromUrl && !isHashy(fromUrl)) return fromUrl;
  const sec = clean(section);
  if (sec && t) return `${sec}（${t}）`;
  return sec || t || `${(url.split(/[?#]/)[0].match(/\.([a-z0-9]+)$/i)?.[1] ?? '附件').toUpperCase()} 檔案`;
}
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
  const hasEmbedded = videos.length > 0;
  for (const l of raw.links ?? []) {
    // 頁面已有嵌入的影片時，「下載影片」連結只是同一部影片的檔案，不另列
    if (hasEmbedded && /download|下載/i.test(l.text)) continue;
    addVideo(l.href, GENERIC_TEXT.test(clean(l.text)) ? titleOf(l.text, l.href, l.section) : l.text, l.section);
  }

  const materials = [];
  const matSeen = new Set();
  for (const l of raw.links ?? []) {
    const direct = MATERIAL_EXT.test(l.href);
    const isResource = RESOURCE_PAGE.test(l.href);
    if ((!direct && !isResource) || matSeen.has(l.href)) continue;
    const text = clean(l.text);
    let ext = extOf(l.href);
    let title;
    if (direct) {
      title = titleOf(text, l.href, l.section);
    } else {
      // 資源頁：翻頁連結、選單、影片頁不算；其餘要有檔案類型提示，或能從文字／區塊歸類才算檔案
      if (l.nav || PAGER.test(text)) continue;
      const hint = (text.match(EXT_HINT) || l.href.match(/_(pdf|pptx?|docx?|xlsx?|zip)\/?$/i) || [])[1];
      title = titleOf(text.replace(EXT_HINT_PAREN, ''), l.href, l.section);
      const cat = classifyMaterial({ title, section: l.section, url: l.href });
      if (!hint && (cat === 'other' || VIDEO_WORDS.test(`${text} ${l.section}`))) continue;
      ext = hint ? hint.toLowerCase() : 'file';
    }
    matSeen.add(l.href);
    materials.push({
      id: l.href, title, url: l.href, ext, section: clean(l.section),
      resolve: isResource && !direct, // 資源頁：點開時再取得實際的檔案網址
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
    if (l.nav || hostOf(u) !== host || !text || text.length > 60) continue;
    if (MATERIAL_EXT.test(u) || RESOURCE_PAGE.test(u) || videoFrom(u) || u.replace(/\/+$/, '') === here || relSeen.has(u)) continue;
    relSeen.add(u);
    related.push({ id: u, title: text, url: u });
    if (related.length >= 40) break;
  }

  const outline = (raw.headings ?? []).map((h) => ({ level: h.level, text: clean(h.text) })).filter((h) => h.text);
  // 太短的多半是按鈕或選單文字；中文資訊密度高，門檻訂為 8 個字以保留「授課教師：王大明」這類短資訊
const overview = [...new Set((raw.paras ?? []).map(clean).filter((s) => s.length >= 8 && !/[«»]/.test(s) && !/^(previous|next)\b/i.test(s)))].slice(0, 12);

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
// hideRelated：多章節課程已有章節分頁，相關頁面只會重複選單
export function tabsOf(content, { hideRelated = false } = {}) {
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
  if (content.related.length && !hideRelated) tabs.push({ key: 'related', label: '相關頁面', count: content.related.length });
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
