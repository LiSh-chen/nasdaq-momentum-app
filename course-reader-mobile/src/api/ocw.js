// 解析 MIT OpenCourseWare 課程首頁的章節與單元清單
import { decodeEntities as decode } from '../text.js';

const textOf = (html) => decode(html.replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();

const humanize = (seg) => seg.replace(/-/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

export function sameUrl(a, b) {
  const norm = (u) => u.split('#')[0].replace(/\/+$/, '');
  return norm(a) === norm(b);
}

export function parseOutline(html, courseUrl) {
  const base = new URL(courseUrl);
  const root = base.pathname.replace(/\/+$/, '');
  const re = /<a\b[^>]*?href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
  const sections = new Map();
  const seen = new Set();

  for (const m of html.matchAll(re)) {
    let url;
    try {
      url = new URL(decode(m[1]), base);
    } catch {
      continue;
    }
    url.hash = '';
    url.search = '';
    if (url.hostname !== base.hostname || !url.pathname.startsWith(`${root}/`)) continue;

    const rest = url.pathname.slice(root.length + 1).split('/').filter(Boolean);
    let key;
    let isHeader;
    if (rest[0] === 'pages' && rest.length >= 2) {
      key = rest[1];
      isHeader = rest.length === 2;
    } else if (rest[0] === 'video_galleries' && rest.length >= 2) {
      key = 'video_galleries';
      isHeader = false;
    } else continue;

    const title = textOf(m[2]);
    const href = url.toString();
    if (!title || seen.has(href)) continue;
    seen.add(href);

    if (!sections.has(key)) {
      sections.set(key, { key, title: key === 'video_galleries' ? '影片' : humanize(key), url: null, items: [] });
    }
    const sec = sections.get(key);
    if (isHeader) {
      sec.title = title;
      sec.url = href;
    } else {
      sec.items.push({ title, url: href });
    }
  }
  return [...sections.values()];
}

// 耶魯開放課程：課程頁上各講次連結是 /系所/課號/lecture-N，與分頁標籤（quicktabs）、檔案區分開
export function parseYaleOutline(html, courseUrl) {
  const base = new URL(courseUrl);
  const root = base.pathname.replace(/\/+$/, '');
  const items = [];
  const seen = new Set();
  for (const m of html.matchAll(/<a\b[^>]*?href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)) {
    let url;
    try { url = new URL(decode(m[1]), base); } catch { continue; }
    url.hash = '';
    url.search = '';
    if (url.hostname !== base.hostname || !url.pathname.startsWith(`${root}/`)) continue;
    const segs = url.pathname.slice(root.length + 1).split('/').filter(Boolean);
    if (segs.length !== 1 || segs[0].includes('.')) continue; // 只要正下一層、不是檔案
    const title = textOf(m[2]);
    const href = url.toString();
    if (!title || seen.has(href)) continue;
    seen.add(href);
    items.push({ title, url: href });
  }
  return items.length ? [{ key: 'lectures', title: '課程講次', url: null, items }] : [];
}

// 攤平成閱讀順序：章節首頁（若有）接著底下的單元
export function flattenOutline(sections) {
  const pages = [];
  for (const s of sections) {
    if (s.url) pages.push({ title: s.title, url: s.url, section: s.title });
    for (const it of s.items) pages.push({ ...it, section: s.title });
  }
  return pages;
}

const cache = new Map();

export async function fetchOutline(courseUrl, fetchImpl = fetch) {
  if (cache.has(courseUrl)) return cache.get(courseUrl);
  const res = await fetchImpl(courseUrl);
  if (!res.ok) throw new Error(`OCW ${res.status}`);
  const parse = new URL(courseUrl).hostname === 'oyc.yale.edu' ? parseYaleOutline : parseOutline;
  const sections = parse(await res.text(), courseUrl);
  cache.set(courseUrl, sections);
  return sections;
}
