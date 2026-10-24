import { interleave, searchCatalog } from './search';
import { searchCourses } from './mitLearn';
import { searchYale } from './yale';
import { searchNycu } from './nycu';

jest.mock('./mitLearn', () => ({ ...jest.requireActual('./mitLearn'), searchCourses: jest.fn() }));
jest.mock('./yale', () => ({ searchYale: jest.fn() }));
jest.mock('./nycu', () => ({ searchNycu: jest.fn() }));

const c = (id) => ({ id, title: id });
const res = (ids, extra = {}) => ({ items: ids.map(c), count: ids.length, next: null, facets: {}, ...extra });

beforeEach(() => {
  jest.resetAllMocks();
  searchCourses.mockResolvedValue({ items: ['m1', 'm2', 'm3'].map(c), count: 2596, hasMore: true, facets: { level: [] } });
  searchYale.mockResolvedValue(res(['y1']));
  searchNycu.mockResolvedValue(res(['n1', 'n2'], { next: 2 }));
});

test('interleave 輪流排列各來源', () => {
  expect(interleave([[1, 2, 3], ['a'], ['x', 'y']])).toEqual([1, 'a', 'x', 2, 'y', 3]);
  expect(interleave([[], []])).toEqual([]);
});

test('MIT：帶入篩選條件與 offset，並以 offset 作為下一頁位置', async () => {
  const r = await searchCatalog({ provider: 'mit', q: 'physics', level: 'graduate', cursor: 20 });
  expect(searchCourses).toHaveBeenCalledWith({ q: 'physics', level: 'graduate', offset: 20 });
  expect(r.next).toBe(23);
  expect(r.facets).toEqual({ level: [] });
});

test('耶魯與陽明交大各自查詢自己的來源', async () => {
  expect((await searchCatalog({ provider: 'yale', q: 'astro' })).items.map((i) => i.id)).toEqual(['y1']);
  expect(searchCourses).not.toHaveBeenCalled();
  const r = await searchCatalog({ provider: 'nycu', q: '微積分', cursor: 2 });
  expect(searchNycu).toHaveBeenCalledWith('微積分', 2);
  expect(r.next).toBe(2);
});

test('全部＋有關鍵字：第一頁混合各來源，總數相加', async () => {
  const r = await searchCatalog({ provider: 'all', q: 'calculus' });
  expect(r.items.map((i) => i.id)).toEqual(['m1', 'y1', 'n1', 'm2', 'n2', 'm3']);
  expect(r.count).toBe(2596 + 1 + 2);
  expect(r.next).toBe(3); // 之後只延續 MIT
});

test('全部：沒有關鍵字或載入更多時只查 MIT', async () => {
  await searchCatalog({ provider: 'all', q: '' });
  await searchCatalog({ provider: 'all', q: 'calculus', cursor: 20 });
  expect(searchYale).not.toHaveBeenCalled();
  expect(searchNycu).not.toHaveBeenCalled();
});

test('全部：某個來源失敗不影響其他來源，並回報失敗數', async () => {
  searchYale.mockRejectedValue(new Error('down'));
  const r = await searchCatalog({ provider: 'all', q: 'calculus' });
  expect(r.items.map((i) => i.id)).toEqual(['m1', 'n1', 'm2', 'n2', 'm3']);
  expect(r.failed).toBe(1);
});
