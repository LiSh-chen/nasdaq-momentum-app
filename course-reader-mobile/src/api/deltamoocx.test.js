import { filterDelta, parseDeltaCard, parseDeltaHome, searchDeltamoocx } from './deltamoocx';

const B = 'https://univ.deltamoocx.net';
const L = (id, text, extra = {}) => ({ href: `${B}/course/preview/${id}`, text, section: '大學自動化學程', image: `https://x/${id}.jpg`, nav: false, ...extra });

// 文字格式取自台達磨課師首頁的實際診斷資料
const RAW = {
  links: [
    { href: `${B}/`, text: '探索', nav: true },
    L('123e7c8480010', '18 小時數位影像處理DDeltaMOOCx 課程管理員2026-03-192k83'),
    L('123e7c8480010', '18 小時數位影像處理DDeltaMOOCx 課程管理員2026-03-192k83', { section: '趨勢' }),
    L('123e7b6a80004', '9 小時108普通型高中數學高二B類第四冊DDeltaMOOCx 課程管理員2026-10-151k33', { section: '本週最佳' }),
    L('16afdb1810001', '4 小時30 分鐘108普通型高中高一加深加廣統整與補救課程DDeltaMOOCx 課程管理員2026-09-2245625', { section: '趨勢' }),
    L('152c121230001', '18 小時微處理機-題庫(因材網授權)DDeltaMOOCx 課程管理員2026-09-211k41', { image: `${B}/images/course_default_169.svg` }),
    { href: `${B}/contact`, text: '聯絡我們', nav: true },
  ],
};

test('解析卡片文字：時數、課名、日期', () => {
  expect(parseDeltaCard('18 小時數位影像處理DDeltaMOOCx 課程管理員2026-03-192k83')).toEqual({ title: '數位影像處理', duration: '18 小時', date: '2026-03-19' });
  expect(parseDeltaCard('4 小時30 分鐘108普通型高中高一加深加廣統整與補救課程DDeltaMOOCx 課程管理員2026-09-2245625')).toMatchObject({ title: '108普通型高中高一加深加廣統整與補救課程', duration: '4 小時30 分鐘' });
  expect(parseDeltaCard('聯絡我們')).toBeNull();
});

test('首頁課程去重，預設封面不當作圖片', () => {
  const r = parseDeltaHome(RAW);
  expect(r.items.map((c) => c.title)).toEqual(['數位影像處理', '108普通型高中數學高二B類第四冊', '108普通型高中高一加深加廣統整與補救課程', '微處理機-題庫(因材網授權)']);
  expect(r.items[0]).toMatchObject({ id: 'deltamoocx:123e7c8480010', url: `${B}/course/preview/123e7c8480010`, sourceName: '台達磨課師', topics: ['大學自動化學程'] });
  expect(r.items[3].image).toBeNull();
});

test('關鍵字篩選', () => {
  const { items } = parseDeltaHome(RAW);
  expect(filterDelta(items, '數學').map((c) => c.title)).toEqual(['108普通型高中數學高二B類第四冊']);
  expect(filterDelta(items, '')).toHaveLength(4);
  expect(filterDelta(items, '不存在')).toHaveLength(0);
});

test('searchDeltamoocx 透過瀏覽器載入首頁', async () => {
  const browser = { run: jest.fn().mockResolvedValue(RAW) };
  const r = await searchDeltamoocx('控制', browser);
  expect(browser.run.mock.calls[0][0]).toBe(`${B}/`);
  expect(browser.run.mock.calls[0][1].ready(RAW)).toBe(true);
  expect(browser.run.mock.calls[0][1].ready({ links: [] })).toBe(false);
  expect(r.count).toBe(0);
  expect((await searchDeltamoocx('', browser)).browse).toBe(true);
});
