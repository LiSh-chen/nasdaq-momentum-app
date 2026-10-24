import { searchCourses } from './mitLearn.js';
import { searchDeltamoocx } from './deltamoocx.js';
import { searchNtu } from './ntu.js';
import { searchNycu } from './nycu.js';
import { searchYale } from './yale.js';

export const PROVIDERS = [
  { id: 'all', label: '全部' },
  { id: 'mit', label: 'MIT' },
  { id: 'yale', label: '耶魯' },
  { id: 'nycu', label: '陽明交大' },
  { id: 'ntu', label: '臺大' },
  { id: 'deltamoocx', label: '台達' },
];

// 各來源的結果輪流排列，讓每個來源的課程都看得到
export function interleave(lists) {
  const out = [];
  const max = Math.max(0, ...lists.map((l) => l.length));
  for (let i = 0; i < max; i++) for (const l of lists) if (i < l.length) out.push(l[i]);
  return out;
}

const settle = async (promise) => {
  try { return await promise; } catch { return null; }
};

// 統一的搜尋入口。cursor 是各來源自己的分頁位置（MIT 為 offset、陽明交大為頁碼）。
// browser：需要真實瀏覽器才能讀取的來源（臺大）使用。「全部」不含這類來源，以免拖慢搜尋。
export async function searchCatalog({ provider = 'all', q = '', cursor, browser, ...filters } = {}) {
  if (provider === 'mit') {
    const r = await searchCourses({ q, ...filters, offset: cursor ?? 0 });
    return { ...r, next: r.hasMore ? (cursor ?? 0) + r.items.length : null };
  }
  if (provider === 'yale') return searchYale(q);
  if (provider === 'nycu') return searchNycu(q, cursor || 1);
  if (provider === 'deltamoocx') return searchDeltamoocx(q, browser);
  if (provider === 'ntu') return searchNtu(q, cursor || 1, browser);

  // 全部：只有第一頁會混入其他來源；之後的分頁（載入更多）只延續 MIT
  const mit = await searchCourses({ q, offset: cursor ?? 0 });
  const mitNext = mit.hasMore ? (cursor ?? 0) + mit.items.length : null;
  if (cursor || !q.trim()) return { ...mit, next: mitNext, facets: {} };

  const [yale, nycu] = await Promise.all([settle(searchYale(q)), settle(searchNycu(q, 1))]);
  const others = [yale, nycu].filter(Boolean);
  return {
    items: interleave([mit.items, ...others.map((o) => o.items)]),
    count: mit.count + others.reduce((n, o) => n + o.count, 0),
    next: mitNext,
    facets: {},
    // 其他來源失敗時不影響 MIT 的結果
    failed: [yale, nycu].filter((x) => x === null).length,
  };
}
