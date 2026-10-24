// 臺大開放式課程（ocw.aca.ntu.edu.tw）。網站有人機驗證，必須由 App 內建瀏覽器載入頁面再讀取結果。
// 搜尋網址與結果結構取自該網站「全部課程」頁的實際診斷資料。
const BASE = 'https://ocw.aca.ntu.edu.tw';

export const buildNtuSearchUrl = (q, page = 1) =>
  `${BASE}/courses?category_id=0&college=&mooc=&search=${encodeURIComponent(q.trim())}&sort_by=desc&grid_type=grid&page=${page}`;

const COURSE_PATH = /^\/courses\/([^/?#]+)\/?$/;
const UNIT_PATH = /^\/courses\/([^/?#]+)\/\d+\/?$/;
// 單元命中：「單元 <單元名稱> 課名: <課名> 查看完整課程」
const UNIT_TEXT = /^單元\s+([\s\S]*?)\s+課名:\s*([\s\S]*?)\s*查看完整課程$/;
const OCW_SUFFIX = /\s*(MOOC)?OCW$/;

const pathOf = (href) => { try { const u = new URL(href); return u.origin === BASE ? u.pathname : ''; } catch { return ''; } };

export function parseNtuResults(raw, page = 1) {
  const links = raw.links ?? [];
  const images = new Map();
  const courses = new Map(); // 課程代碼 → 課程
  const units = new Map();   // 課程代碼 → { name, titles }

  for (const l of links) {
    const code = pathOf(l.href).match(COURSE_PATH)?.[1];
    if (code && l.image && !images.has(code)) images.set(code, l.image);
  }

  for (const l of links) {
    if (l.nav) continue;
    const path = pathOf(l.href);
    const text = (l.text ?? '').replace(/\s+/g, ' ').trim();
    const course = path.match(COURSE_PATH)?.[1];
    if (course && text) {
      if (!courses.has(course)) courses.set(course, text.replace(OCW_SUFFIX, '').trim());
      continue;
    }
    const unit = path.match(UNIT_PATH)?.[1];
    const m = unit && text.match(UNIT_TEXT);
    if (m) {
      const u = units.get(unit) ?? { name: m[2], titles: [] };
      u.titles.push(m[1]);
      units.set(unit, u);
    }
  }
  for (const [code, u] of units) if (!courses.has(code)) courses.set(code, u.name); // 只有單元命中的課程

  const items = [...courses].map(([code, title]) => {
    const u = units.get(code);
    const shown = u ? u.titles.slice(0, 3).join('、') : '';
    return {
      id: `ntu:${code}`,
      title,
      description: u ? `符合的單元：${shown}${u.titles.length > 3 ? ` 等 ${u.titles.length} 個單元` : ''}` : '',
      image: images.get(code) ?? null,
      url: `${BASE}/courses/${code}`,
      sourceCode: 'ntu',
      sourceName: '臺大開放式課程',
      topics: [],
      levels: [],
      instructors: [],
      features: [],
      free: true,
      requiresLogin: false,
      unified: false,
    };
  });

  const nextRe = new RegExp(`[?&]page=${page + 1}(?!\\d)`);
  const next = links.some((l) => pathOf(l.href) === '/courses' && nextRe.test(l.href)) ? page + 1 : null;
  return { items, count: items.length, next, facets: {} };
}

// browser：{ run(url, { ready }) → 擷取結果 }，由 App 內的隱藏瀏覽器提供
export async function searchNtu(query, page = 1, browser) {
  if (!String(query ?? '').trim()) return { items: [], count: 0, next: null, facets: {} };
  if (!browser) throw new Error('browser required');
  const raw = await browser.run(buildNtuSearchUrl(query, page), {
    // 搜尋結果是頁面載入後才產生的：有課程連結，或明確顯示查無結果，才算完成
    ready: (msg) => parseNtuResults(msg, page).items.length > 0 || (msg.paras ?? []).some((p) => /找不到|查無|沒有符合|無符合/.test(p)),
  });
  return parseNtuResults(raw, page);
}
