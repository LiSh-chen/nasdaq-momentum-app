import { decodeEntities, stripHtml } from '../text.js';
import { getText } from './http.js';

// 陽明交通大學開放式課程（WordPress）。搜尋結果頁 ?s=關鍵字，分頁 ?paged=N&s=關鍵字
const BASE = 'https://ocw.nycu.edu.tw/';

export const buildNycuSearchUrl = (q, page = 1) =>
  `${BASE}?${page > 1 ? `paged=${page}&` : ''}s=${encodeURIComponent(q.trim())}`;

// 圖片網址含未編碼的中文；已編碼的也不會被重複編碼
const safeEncode = (u) => { try { return encodeURI(decodeURI(u)); } catch { return encodeURI(u); } };

export function parseNycuResults(html) {
  const items = [];
  const seen = new Set();
  for (const m of html.matchAll(/<article\b[^>]*class="[^"]*\bcourse_page\b[^"]*"[^>]*>([\s\S]*?)<\/article>/g)) {
    const block = m[1];
    const a = block.match(/<h3[^>]*elementor-post__title[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
    if (!a) continue;
    const url = decodeEntities(a[1]).trim();
    if (seen.has(url)) continue;
    seen.add(url);
    const full = stripHtml(a[2]);
    const bar = full.lastIndexOf('|');
    const title = (bar > 0 ? full.slice(0, bar) : full).trim();
    const meta = bar > 0 ? full.slice(bar + 1).trim() : '';
    const image = block.match(/data-src="(https?:[^"]+)"/)?.[1] ?? block.match(/<img[^>]*\ssrc="(https?:[^"]+)"/)?.[1] ?? null;
    const date = stripHtml(block.match(/elementor-post-date[^>]*>([\s\S]*?)<\/span>/)?.[1] ?? '');
    const excerpt = stripHtml(block.match(/elementor-post__excerpt[^>]*>([\s\S]*?)<\/div>/)?.[1] ?? '');
    let slug = url;
    try { slug = decodeURIComponent(url.split('course_page=all-course/')[1] ?? url); } catch {}
    items.push({
      id: `nycu:${slug}`,
      title,
      description: [excerpt, date && `發布：${date}`].filter(Boolean).join('\n'),
      image: image ? safeEncode(image) : null,
      url,
      sourceCode: 'nycu',
      sourceName: '陽明交大 OCW',
      topics: meta ? [meta] : [],
      levels: [],
      instructors: [],
      features: [],
      free: true,
      requiresLogin: false,
      unified: false,
    });
  }
  return items;
}

export async function searchNycu(query, page, fetchImpl = fetch) {
  if (!String(query ?? '').trim()) return { items: [], count: 0, next: null, facets: {} };
  const p = page || 1; // 0 或未提供都是第 1 頁
  const url = buildNycuSearchUrl(query, p);
  const html = await getText(url, fetchImpl);
  const items = parseNycuResults(html);
  const hasMore = html.includes(`paged=${p + 1}`);
  return { items, count: items.length, next: hasMore ? p + 1 : null, facets: {}, meta: { url, bytes: html.length, parsed: items.length } };
}
