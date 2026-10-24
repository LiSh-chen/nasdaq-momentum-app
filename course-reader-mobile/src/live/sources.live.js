/** @jest-environment node */
// 對真實的耶魯／陽明交大網站驗證搜尋解析（需要網路）
import { searchYale, resetYaleCache } from '../api/yale';
import { searchNycu, parseNycuResults } from '../api/nycu';
import { getText } from '../api/http';
import { searchCatalog } from '../api/search';

jest.setTimeout(120000);

test('耶魯：讀得到完整課程列表，可依關鍵字篩選', async () => {
  resetYaleCache();
  const all = await searchYale('');
  console.log(`yale: ${all.count} courses; first: ${JSON.stringify(all.items[0])}`);
  expect(all.count).toBeGreaterThanOrEqual(30);
  expect(all.items.every((c) => c.title && c.url.startsWith('https://oyc.yale.edu/') && c.instructors.length)).toBe(true);
  const astro = await searchYale('astro');
  expect(astro.items.some((c) => /astr-160/.test(c.url))).toBe(true);
});

test('陽明交大：搜尋「微積分」得到課程卡片，含標題、連結與縮圖，並能翻頁', async () => {
  const p1 = await searchNycu('微積分', 1);
  console.log(`nycu p1: ${p1.items.length} items next=${p1.next}; first: ${JSON.stringify({ ...p1.items[0], description: (p1.items[0]?.description || '').slice(0, 60) })}`);
  expect(p1.items.length).toBeGreaterThanOrEqual(5);
  expect(p1.items.every((c) => c.title && c.url.includes('course_page=all-course/'))).toBe(true);
  expect(p1.items.some((c) => c.title.includes('微積分'))).toBe(true);
  expect(p1.items.filter((c) => c.image).length).toBeGreaterThan(0);
  if (p1.next) {
    const p2 = await searchNycu('微積分', p1.next);
    console.log(`nycu p2: ${p2.items.length} items; first: ${p2.items[0]?.title}`);
    expect(p2.items.length).toBeGreaterThan(0);
    expect(p2.items[0].url).not.toBe(p1.items[0].url);
  }
});

test('全部：同一個關鍵字混合 MIT、耶魯、陽明交大', async () => {
  const r = await searchCatalog({ provider: 'all', q: 'calculus' });
  const bySource = r.items.reduce((m, c) => ({ ...m, [c.sourceCode]: (m[c.sourceCode] ?? 0) + 1 }), {});
  console.log(`all/calculus: ${r.items.length} items by source ${JSON.stringify(bySource)} total=${r.count} failed=${r.failed}`);
  expect(r.items.length).toBeGreaterThan(5);
  expect(r.failed).toBe(0);
  expect(Object.keys(bySource).length).toBeGreaterThanOrEqual(2);
});

test('陽明交大：哪些頁面不用關鍵字也能列出課程（探測，僅輸出資訊）', async () => {
  const urls = [
    'https://ocw.nycu.edu.tw/',
    'https://ocw.nycu.edu.tw/?s=',
    'https://ocw.nycu.edu.tw/?post_type=course_page',
    'https://ocw.nycu.edu.tw/?post_type=course_page&paged=2',
    'https://ocw.nycu.edu.tw/?course_page=all-course',
    'https://ocw.nycu.edu.tw/?course_page=all-course/college-of-science',
  ];
  for (const url of urls) {
    try {
      const html = await getText(url);
      const items = parseNycuResults(html);
      const articles = (html.match(/<article\b/g) || []).length;
      const courseLinks = new Set((html.match(/course_page=all-course\/[^"'&<>\s]+/g) || [])).size;
      const pages = [...new Set((html.match(/paged=\d+/g) || []))].slice(0, 6);
      console.log(`[${url}] bytes=${html.length} articles=${articles} parsed=${items.length} courseLinks=${courseLinks} pager=${pages.join(',')} first=${items.slice(0, 3).map((c) => c.title).join(' ; ')}`);
    } catch (e) {
      console.log(`[${url}] ERROR ${e.message}`);
    }
  }
});
