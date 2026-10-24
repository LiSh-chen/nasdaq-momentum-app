import { decodeEntities, stripHtml } from '../text.js';
import { getText } from './http.js';

// Open Yale Courses（CC 授權）。課程列表是一張表格，只有約 40 門課，整張讀下來在本機篩選。
const BASE = 'https://oyc.yale.edu';
const LIST_URL = `${BASE}/courses`;
const TTL = 60 * 60 * 1000;

const cell = (row, name) =>
  row.match(new RegExp(`<td[^>]*class="[^"]*\\bviews-field-${name}(?![\\w-])[^"]*"[^>]*>([\\s\\S]*?)</td>`))?.[1] ?? '';

export function parseYaleListing(html) {
  const out = [];
  const seen = new Set();
  for (const m of html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)) {
    const row = m[1];
    const titleCell = cell(row, 'title-1');
    const href = titleCell.match(/href="([^"]+)"/)?.[1];
    const title = stripHtml(titleCell);
    if (!href || !title || seen.has(href)) continue;
    seen.add(href);
    out.push({
      title,
      url: new URL(decodeEntities(href), BASE).toString(),
      dept: stripHtml(cell(row, 'title')),
      code: stripHtml(cell(row, 'field-course-number')),
      professor: stripHtml(cell(row, 'field-course-professor-last-name')),
      semester: stripHtml(cell(row, 'field-semester')),
    });
  }
  return out;
}

export function toCourse(c) {
  return {
    id: `yale:${new URL(c.url).pathname.replace(/^\/|\/$/g, '')}`,
    title: c.title,
    description: [c.dept && `系所：${c.dept}`, c.code && `課號：${c.code}`, c.semester && `開課學期：${c.semester}`].filter(Boolean).join('　'),
    image: null,
    url: c.url,
    sourceCode: 'yale',
    sourceName: 'Open Yale Courses',
    topics: c.dept ? [c.dept] : [],
    levels: [],
    instructors: c.professor ? [c.professor] : [],
    features: ['Lecture Videos'],
    free: true,
    requiresLogin: false,
    unified: false,
  };
}

let cache = null;

export async function searchYale(query, fetchImpl = fetch, now = Date.now()) {
  if (!cache || now - cache.at > TTL) {
    cache = { at: now, items: parseYaleListing(await getText(LIST_URL, fetchImpl)).map(toCourse) };
  }
  const q = String(query ?? '').trim().toLowerCase();
  const items = q
    ? cache.items.filter((c) => `${c.title} ${c.topics.join(' ')} ${c.instructors.join(' ')} ${c.description}`.toLowerCase().includes(q))
    : cache.items;
  return { items, count: items.length, next: null, facets: {} };
}

export const resetYaleCache = () => { cache = null; };
