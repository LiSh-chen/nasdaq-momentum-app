// 台達磨課師（univ.deltamoocx.net）。網站是前端動態產生的頁面，必須由 App 內建瀏覽器載入後再讀取。
// 首頁的課程卡片結構取自該網站首頁的實際診斷資料；搜尋框沒有可直接呼叫的網址，所以先取得首頁課程再依關鍵字篩選。
const BASE = 'https://univ.deltamoocx.net';
const COURSE_PATH = /^\/course\/preview\/([^/?#]+)\/?$/;
// 卡片文字：「<時數><課名>D<頻道名> <日期><觀看數與讚數>」，例如「18 小時數位影像處理DDeltaMOOCx 課程管理員2026-03-192k83」
const CARD_TEXT = /^(.*?)DDeltaMOOCx 課程管理員(\d{4}-\d{2}-\d{2})/;
const DURATION = /^((?:\d+\s*小時)?\s*(?:\d+\s*分鐘)?)/;

const pathOf = (href) => { try { const u = new URL(href); return u.origin === BASE ? u.pathname : ''; } catch { return ''; } };

export function parseDeltaCard(text) {
  const m = String(text ?? '').replace(/\s+/g, ' ').trim().match(CARD_TEXT);
  if (!m) return null;
  const duration = m[1].match(DURATION)[1].trim();
  const title = m[1].slice(m[1].match(DURATION)[1].length).trim();
  return title ? { title, duration, date: m[2] } : null;
}

export function parseDeltaHome(raw) {
  const seen = new Map();
  for (const l of raw.links ?? []) {
    if (l.nav) continue;
    const id = pathOf(l.href).match(COURSE_PATH)?.[1];
    if (!id || seen.has(id)) continue;
    const card = parseDeltaCard(l.text);
    if (!card) continue;
    const hasCover = l.image && !/course_default/.test(l.image);
    seen.set(id, {
      id: `deltamoocx:${id}`,
      title: card.title,
      description: [card.duration && `課程時數 ${card.duration}`, `發布 ${card.date}`].filter(Boolean).join('　'),
      image: hasCover ? l.image : null,
      url: `${BASE}/course/preview/${id}`,
      sourceCode: 'deltamoocx',
      sourceName: '台達磨課師',
      topics: [l.section].filter((s) => s && !/本週最佳|最新發布|趨勢/.test(s)),
      levels: [],
      instructors: [],
      features: [],
      free: true,
      requiresLogin: false,
      unified: false,
    });
  }
  const items = [...seen.values()];
  return { items, count: items.length, next: null, facets: {} };
}

export const filterDelta = (items, q) => {
  const words = String(q ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  return words.length ? items.filter((c) => words.every((w) => `${c.title} ${c.topics.join(' ')}`.toLowerCase().includes(w))) : items;
};

// browser：{ run(url, { ready }) → 擷取結果 }，由 App 內的隱藏瀏覽器提供
export async function searchDeltamoocx(query, browser) {
  if (!browser) throw new Error('browser required');
  const url = `${BASE}/`;
  const raw = await browser.run(url, { ready: (msg) => parseDeltaHome(msg).items.length > 0 });
  const all = parseDeltaHome(raw);
  const items = filterDelta(all.items, query);
  return { items, count: items.length, next: null, facets: {}, browse: !String(query ?? '').trim(), meta: { url, parsed: all.items.length, matched: items.length } };
}
