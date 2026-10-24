import { stripHtml } from '../text.js';

// MIT Learn 公開 API：彙整 MIT OpenCourseWare、MITx 等來源的課程目錄，不需金鑰
const BASE = 'https://api.learn.mit.edu/api/v1';
export const PAGE_SIZE = 20;

export const SOURCE_NAMES = {
  ocw: 'MIT OpenCourseWare',
  mitx: 'MITx',
  xpro: 'MIT xPRO',
  mitpe: 'MIT Professional Education',
  see: 'MIT SEE',
};

export const LEVEL_NAMES = {
  undergraduate: '大學部',
  graduate: '研究所',
  high_school: '高中',
  noncredit: '非學分',
};


export function buildSearchUrl({ q, offeredBy, level, topic, hasVideos, sort, offset = 0, limit = PAGE_SIZE } = {}) {
  const p = new URLSearchParams();
  p.append('resource_type', 'course');
  if (q && q.trim()) p.append('q', q.trim());
  if (offeredBy) p.append('offered_by', offeredBy);
  if (level) p.append('level', level);
  if (topic) p.append('topic', topic);
  if (hasVideos) p.append('course_feature', 'Lecture Videos');
  if (sort) p.append('sortby', sort);
  for (const agg of ['offered_by', 'level', 'topic']) p.append('aggregations', agg);
  p.append('limit', String(limit));
  p.append('offset', String(offset));
  return `${BASE}/learning_resources_search/?${p.toString()}`;
}

export function normalizeCourse(r) {
  const run = r.runs?.[0] ?? {};
  const code = r.offered_by?.code ?? '';
  return {
    id: r.id,
    title: r.title ?? '',
    description: stripHtml(r.description),
    image: r.image?.url ?? null,
    url: r.url,
    sourceCode: code,
    sourceName: r.offered_by?.name || SOURCE_NAMES[code] || code.toUpperCase(),
    topics: (r.topics ?? []).map((t) => t.name),
    levels: (run.level ?? []).map((l) => l.name),
    instructors: (run.instructors ?? []).map((i) => i.full_name).filter(Boolean),
    features: r.course_feature ?? [],
    free: r.free !== false,
    requiresLogin: code !== 'ocw',
    // 開放授權（CC）的 OCW 課程才用統一介面閱讀；其餘來源需登入，改用 App 內網頁
    unified: code === 'ocw',
  };
}

export async function searchCourses(params, fetchImpl = fetch) {
  const res = await fetchImpl(buildSearchUrl(params));
  if (!res.ok) throw new Error(`MIT Learn API ${res.status}`);
  const json = await res.json();
  const aggs = json.metadata?.aggregations ?? {};
  const facet = (name) => (aggs[name] ?? []).map((f) => ({ key: f.key, count: f.doc_count }));
  return {
    items: (json.results ?? []).map(normalizeCourse),
    count: json.count ?? 0,
    hasMore: Boolean(json.next),
    facets: { offered_by: facet('offered_by'), level: facet('level'), topic: facet('topic') },
  };
}
