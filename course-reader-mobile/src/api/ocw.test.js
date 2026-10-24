import { fetchOutline, flattenOutline, parseOutline, parseYaleOutline, sameUrl } from './ocw';

const COURSE = 'https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/';
const R = '/courses/8-01sc-classical-mechanics-fall-2016';

// 連結格式取自 OCW 課程首頁的實際內容（文字前後有空白、含 HTML 實體）
const HTML = `
<a href="${R}/pages/syllabus/">
  Syllabus
</a>
<a href="${R}/pages/syllabus/about-the-team/">About the Team</a>
<a href="${R}/pages/week-1-kinematics/">Week 1: Kinematics</a>
<a href="${R}/pages/week-1-kinematics/1-1-coordinate-systems/"><span>Lesson 1: 1D Kinematics &amp; Position</span></a>
<a href="${R}/pages/week-1-kinematics/problem-set-1/">Problem Set 1</a>
<a href="${R}/pages/week-1-kinematics/problem-set-1/">Problem Set 1</a>
<a href="${R}/pages/week-2-newtons-laws/problem-set-2/">Problem Set 2</a>
<a href="${R}/video_galleries/lecture-videos/">Lecture Videos</a>
<a href="${R}/resources/0-1/">資源檔（不納入目錄）</a>
<a href="${R}/download">Download</a>
<a href="/courses/other-course/pages/x/">別的課程</a>
<a href="https://example.com/pages/x/">外部</a>
<a href="${R}/pages/empty-title/">   </a>
`;

test('parseOutline 依章節分組並保留順序', () => {
  const s = parseOutline(HTML, COURSE);
  expect(s.map((x) => x.key)).toEqual(['syllabus', 'week-1-kinematics', 'week-2-newtons-laws', 'video_galleries']);
  expect(s[0]).toMatchObject({ title: 'Syllabus', items: [{ title: 'About the Team' }] });
  expect(s[1].title).toBe('Week 1: Kinematics');
  expect(s[1].items.map((i) => i.title)).toEqual(['Lesson 1: 1D Kinematics & Position', 'Problem Set 1']);
  expect(s[2].title).toBe('Week 2 newtons laws'); // 沒有章節首頁連結時由路徑推算標題
  expect(s[3]).toMatchObject({ title: '影片', items: [{ title: 'Lecture Videos' }] });
});

test('parseOutline 排除資源檔、下載連結、其他課程與外部網站', () => {
  const urls = flattenOutline(parseOutline(HTML, COURSE)).map((p) => p.url);
  expect(urls.every((u) => u.startsWith(`https://ocw.mit.edu${R}/`))).toBe(true);
  expect(urls.some((u) => /resources|download|other-course|empty-title/.test(u))).toBe(false);
  expect(new Set(urls).size).toBe(urls.length);
});

test('flattenOutline 先章節首頁再底下單元', () => {
  const pages = flattenOutline(parseOutline(HTML, COURSE));
  expect(pages.map((p) => p.title).slice(0, 5)).toEqual([
    'Syllabus', 'About the Team', 'Week 1: Kinematics', 'Lesson 1: 1D Kinematics & Position', 'Problem Set 1',
  ]);
  expect(pages[3].section).toBe('Week 1: Kinematics');
});

test('sameUrl 忽略結尾斜線與錨點', () => {
  expect(sameUrl('https://a.b/c/', 'https://a.b/c#top')).toBe(true);
  expect(sameUrl('https://a.b/c', 'https://a.b/d')).toBe(false);
});

test('fetchOutline 會快取同一門課', async () => {
  const fetchImpl = jest.fn().mockResolvedValue({ ok: true, text: async () => HTML });
  const url = 'https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/?cache';
  await fetchOutline(url, fetchImpl);
  await fetchOutline(url, fetchImpl);
  expect(fetchImpl).toHaveBeenCalledTimes(1);
});

describe('耶魯課程的講次', () => {
  const COURSE_Y = 'https://oyc.yale.edu/astronomy/astr-160';
  // 結構取自 oyc.yale.edu 課程頁：分頁標籤 (quicktabs) 與各講次連結
  const YALE = `
  <a href="/quicktabs/nojs/course_quicktabs/0">ABOUT</a><a href="/quicktabs/nojs/course_quicktabs/2">SESSIONS</a>
  <a href="/astronomy">Astronomy</a><a href="/astronomy/astr-160">Frontiers and Controversies in Astrophysics</a>
  <a href="https://oyc.yale.edu/astronomy/astr-160/lecture-1">Introduction</a>
  <a href="/astronomy/astr-160/lecture-2"> Planetary&nbsp;Orbits </a>
  <a href="/astronomy/astr-160/lecture-2">重複</a>
  <a href="/astronomy/astr-160/lecture-10">Exoplanets &amp; Transits</a>
  <a href="/astronomy/astr-160/sites/default/files/paper.pdf">不是講次</a>
  <a href="/astronomy/other-course/lecture-1">別的課</a>
  <a href="/astronomy/astr-160/lecture-3"></a>`;

  test('只取該課程底下的各講次，依出現順序，去重並解碼', () => {
    const s = parseYaleOutline(YALE, COURSE_Y);
    expect(s).toHaveLength(1);
    expect(s[0]).toMatchObject({ key: 'lectures', title: '課程講次', url: null });
    expect(s[0].items.map((i) => i.title)).toEqual(['Introduction', 'Planetary Orbits', 'Exoplanets & Transits']);
    expect(s[0].items[0].url).toBe('https://oyc.yale.edu/astronomy/astr-160/lecture-1');
  });

  test('攤平成閱讀順序，每章都帶有所屬區塊', () => {
    const pages = flattenOutline(parseYaleOutline(YALE, COURSE_Y));
    expect(pages.map((p) => p.title)).toEqual(['Introduction', 'Planetary Orbits', 'Exoplanets & Transits']);
    expect(pages[0].section).toBe('課程講次');
  });

  test('沒有講次時回傳空陣列', () => {
    expect(parseYaleOutline('<a href="/astronomy">Astronomy</a>', COURSE_Y)).toEqual([]);
  });

  test('fetchOutline 依網站選擇解析方式', async () => {
    const f = jest.fn().mockResolvedValue({ ok: true, text: async () => YALE });
    const s = await fetchOutline(COURSE_Y + '?t=1', f);
    expect(s[0].items).toHaveLength(3);
  });
});
