import { buildNycuSearchUrl, parseNycuResults, searchNycu } from './nycu';

const U1 = 'https://ocw.nycu.edu.tw/?course_page=all-course/college-of-science/am/%e5%be%ae%e7%a9%8d%e5%88%86%e4%b8%80-calculus-i-%e6%87%89%e7%94%a8%e6%95%b8%e5%ad%b8%e7%b3%bb-%e8%8e%8a%e9%87%8d%e8%80%81%e5%b8%ab';
const U2 = 'https://ocw.nycu.edu.tw/?course_page=all-course/college-of-science/am/calculus-a-ii-academic-109';

// 結構取自 ocw.nycu.edu.tw 搜尋結果頁（Elementor 文章卡片）
const article = (url, title, img = '', extra = '') => `
<article class="elementor-post elementor-grid-item post-28189 course_page type-course_page status-publish format-standard has-post-thumbnail hentry">
	<a class="elementor-post__thumbnail__link" href="${url}" >
	<div class="elementor-post__thumbnail">${img ? `<img width="949" height="571" alt="" data-src="${img}" class="attachment-full lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAAB">` : ''}</div>
	</a>
	<div class="elementor-post__text">
	<h3 class="elementor-post__title">
	<a href="${url}" >
	${title}	</a>
	</h3>
	<div class="elementor-post__meta-data"><span class="elementor-post-date">2022 年 9 月 12 日</span></div>
	<div class="elementor-post__excerpt"><p>${extra}</p></div>
	<a class="elementor-post__read-more" href="${url}" aria-label="Read more">Read More »</a>
	</div>
</article>`;

const PAGE = `<html><body><div class="posts">
${article(U1, '微積分(一) Calculus I | 應用數學系 莊重老師', 'https://ocw.nycu.edu.tw/wp-content/uploads/2022/09/微積分一-Calculus-I-95學年度.jpg', '本課程介紹&nbsp;極限與導數')}
${article(U2, 'Calculus A-II | Academic 109')}
${article(U1, '重複的卡片')}
</div><a class="page-numbers" href="https://ocw.nycu.edu.tw/?paged=2&#038;s=%E5%BE%AE%E7%A9%8D%E5%88%86">2</a>
<article class="post"><h3 class="elementor-post__title"><a href="https://ocw.nycu.edu.tw/?p=1">不是課程的文章</a></h3></article></body></html>`;

const ok = (html) => jest.fn().mockResolvedValue({ ok: true, text: async () => html });

test('搜尋網址：第一頁與分頁', () => {
  expect(buildNycuSearchUrl(' 微積分 ')).toBe('https://ocw.nycu.edu.tw/?s=%E5%BE%AE%E7%A9%8D%E5%88%86');
  expect(buildNycuSearchUrl('微積分', 2)).toBe('https://ocw.nycu.edu.tw/?paged=2&s=%E5%BE%AE%E7%A9%8D%E5%88%86');
});

test('解析結果卡片：標題、教師／系所、縮圖、說明，且只取課程文章並去重', () => {
  const items = parseNycuResults(PAGE);
  expect(items).toHaveLength(2);
  expect(items[0]).toMatchObject({
    title: '微積分(一) Calculus I', url: U1, topics: ['應用數學系 莊重老師'], sourceCode: 'nycu', sourceName: '陽明交大 OCW',
    free: true, requiresLogin: false,
  });
  expect(items[0].id).toBe('nycu:college-of-science/am/微積分一-calculus-i-應用數學系-莊重老師');
  expect(items[0].image).toBe('https://ocw.nycu.edu.tw/wp-content/uploads/2022/09/%E5%BE%AE%E7%A9%8D%E5%88%86%E4%B8%80-Calculus-I-95%E5%AD%B8%E5%B9%B4%E5%BA%A6.jpg');
  expect(items[0].description).toContain('本課程介紹 極限與導數');
  expect(items[1]).toMatchObject({ title: 'Calculus A-II', topics: ['Academic 109'], image: null });
});

test('searchNycu：有下一頁連結時回傳 next，沒有就是 null', async () => {
  const r = await searchNycu('微積分', 1, ok(PAGE));
  expect(r.items).toHaveLength(2);
  expect(r.next).toBe(2);
  expect((await searchNycu('微積分', 2, ok(PAGE.replace('paged=2', 'paged=9')))).next).toBeNull();
});

test('關鍵字為空時改用預設關鍵字「微積分」搜尋，並標示為預設內容', async () => {
  const f = ok(PAGE);
  const r = await searchNycu('  ', 1, f);
  expect(f.mock.calls[0][0]).toBe(buildNycuSearchUrl('微積分', 1));
  expect(r).toMatchObject({ browse: true, defaultQuery: '微積分' });
  expect(r.items).toHaveLength(2);
  expect(r.next).toBe(2); // 預設內容也可以翻頁
});

test('有輸入關鍵字時不是預設內容', async () => {
  const r = await searchNycu('物理', 1, ok(PAGE));
  expect(r.browse).toBeUndefined();
  expect(r.defaultQuery).toBeUndefined();
});

test('頁碼 0 或未提供都視為第 1 頁；下一頁以正確的頁碼判斷', async () => {
  const f = ok(PAGE);
  const r = await searchNycu('微積分', 0, f);
  expect(f.mock.calls[0][0]).toBe('https://ocw.nycu.edu.tw/?s=%E5%BE%AE%E7%A9%8D%E5%88%86');
  expect(r.next).toBe(2);
  expect((await searchNycu('微積分', undefined, ok(PAGE))).next).toBe(2);
});

test('結果帶有查詢資訊（供回報問題）', async () => {
  const r = await searchNycu('微積分', 1, ok(PAGE));
  expect(r.meta).toMatchObject({ url: 'https://ocw.nycu.edu.tw/?s=%E5%BE%AE%E7%A9%8D%E5%88%86', parsed: 2 });
  expect(r.meta.bytes).toBeGreaterThan(100);
});
