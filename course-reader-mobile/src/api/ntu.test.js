import { buildNtuSearchUrl, parseNtuResults, searchNtu } from './ntu';

const B = 'https://ocw.aca.ntu.edu.tw';
const link = (path, text, extra = {}) => ({ href: `${B}${path}`, text, section: '', nav: false, ...extra });
const Q = 'search=%E5%BE%AE%E7%A9%8D%E5%88%86';
const page = (n) => `${B}/courses?category_id=0&college=&mooc=&${Q}&sort_by=desc&grid_type=grid&page=${n}`;

// 結構取自臺大 OCW「全部課程」搜尋「微積分」的實際診斷資料
const RAW = {
  type: 'content', url: `${B}/courses?&category_id=0&college=&mooc=&${Q}&sort_by=desc&grid_type=grid`,
  paras: [], headings: [], frames: [], videos: [], forms: [],
  links: [
    link('/', '', { nav: true }),
    link('/courses', '探索課程', { nav: true }),
    link('/courses', '全部課程', { nav: true }),
    link('/high_school_zones', '高中專區', { nav: true }),
    // 預設的課程方塊：只有圖片，沒有文字，不是搜尋結果
    link('/courses/114S201', '', { image: `${B}/img/114S201.jpg` }),
    link('/courses/114S103', ''),
    link('/courses?grid_type=grid', '清除篩選'),
    // 搜尋結果
    link('/courses/111S205', 'Calculus (4) Kwok-Wing Tsoi (蔡國榮) OCW', { image: `${B}/img/111S205.jpg` }),
    link('/courses/111S102', 'Calculus (1) Kwok-Wing Tsoi (蔡國榮) OCW'),
    link('/courses/103S121', '微積分 (非理工科) 翁秉仁 OCW'),
    link('/courses/103S121/12', '單元 微積分基本定理與積分技巧 - 分部積分法 課名: 微積分 (非理工科) 查看完整課程'),
    link('/courses/102S116', '經濟學原理 林明仁 OCW'),
    link('/courses/102S116/4', '單元 微積分 課名: 經濟學原理 查看完整課程'),
    link('/courses/100S111', '微積分 朱樺 OCW'),
    link('/courses/100S111/58', '單元 第15章 - 向量場 (5) 及 第16章 - 向量微積分 (1) 課名: 微積分 查看完整課程'),
    link('/courses/100S111/59', '單元 第16章 - 向量微積分 (2) 課名: 微積分 查看完整課程'),
    link('/courses/100S111/60', '單元 第16章 - 向量微積分 (3) 課名: 微積分 查看完整課程'),
    link('/courses/100S111/61', '單元 第16章 - 向量微積分 (4) 及 第17章 - 微分方程 (1) 課名: 微積分 查看完整課程'),
    link('/courses/mooc0074', 'AI 時代人類的獨特性 葉素玲 MOOCOCW'),
    link('/courses/999S999/7', '單元 只有單元命中 課名: 孤立的課程 查看完整課程'),
    { href: page(0), text: '', section: '', nav: false },
    { href: page(1), text: '1', section: '', nav: false },
    { href: page(2), text: '', section: '', nav: false },
    link('/about/terms', '使用者條款', { nav: true }),
  ],
};

test('搜尋網址：關鍵字編碼、頁碼', () => {
  expect(buildNtuSearchUrl(' 微積分 ', 1)).toBe(`${B}/courses?category_id=0&college=&mooc=&${Q}&sort_by=desc&grid_type=grid&page=1`);
  expect(buildNtuSearchUrl('微積分', 3)).toContain('&page=3');
});

test('解析搜尋結果：課程名稱去掉「OCW」後綴，沒有文字的圖片方塊與選單不算', () => {
  const { items } = parseNtuResults(RAW, 1);
  expect(items.map((c) => c.title)).toEqual([
    'Calculus (4) Kwok-Wing Tsoi (蔡國榮)', 'Calculus (1) Kwok-Wing Tsoi (蔡國榮)', '微積分 (非理工科) 翁秉仁', '經濟學原理 林明仁', '微積分 朱樺',
    'AI 時代人類的獨特性 葉素玲', '孤立的課程',
  ]);
  expect(items[0]).toMatchObject({
    id: 'ntu:111S205', url: `${B}/courses/111S205`, image: `${B}/img/111S205.jpg`, sourceCode: 'ntu', sourceName: '臺大開放式課程',
    free: true, requiresLogin: false, unified: false,
  });
});

test('命中的單元附在課程下；只有單元命中的課程也會列出', () => {
  const { items } = parseNtuResults(RAW, 1);
  const by = Object.fromEntries(items.map((c) => [c.id, c]));
  expect(by['ntu:100S111'].description).toBe('符合的單元：第15章 - 向量場 (5) 及 第16章 - 向量微積分 (1)、第16章 - 向量微積分 (2)、第16章 - 向量微積分 (3) 等 4 個單元');
  expect(by['ntu:102S116'].description).toBe('符合的單元：微積分');
  expect(by['ntu:111S205'].description).toBe('');
  expect(by['ntu:999S999']).toMatchObject({ title: '孤立的課程', url: `${B}/courses/999S999` });
});

test('下一頁：有 page=N+1 的連結才有', () => {
  expect(parseNtuResults(RAW, 1).next).toBe(2);
  expect(parseNtuResults(RAW, 2).next).toBeNull(); // 沒有 page=3
  expect(parseNtuResults({ ...RAW, links: RAW.links.filter((l) => !/page=2/.test(l.href)) }, 1).next).toBeNull();
});

test('searchNtu：以瀏覽器載入搜尋頁，結果出現才算完成', async () => {
  const run = jest.fn().mockResolvedValue(RAW);
  const r = await searchNtu('微積分', 1, { run });
  expect(run.mock.calls[0][0]).toBe(buildNtuSearchUrl('微積分', 1));
  expect(run.mock.calls[0][1].ready(RAW)).toBe(true);
  expect(run.mock.calls[0][1].ready({ ...RAW, links: [], paras: [] })).toBe(false);
  expect(run.mock.calls[0][1].ready({ ...RAW, links: [], paras: ['找不到符合條件的課程'] })).toBe(true);
  expect(r).toMatchObject({ count: 7, next: 2 });
});

test('關鍵字為空時不載入網頁；沒有瀏覽器時丟出錯誤', async () => {
  const run = jest.fn();
  expect(await searchNtu('  ', 1, { run })).toEqual({ items: [], count: 0, next: null, facets: {} });
  expect(run).not.toHaveBeenCalled();
  await expect(searchNtu('微積分', 1, null)).rejects.toThrow('browser');
});
