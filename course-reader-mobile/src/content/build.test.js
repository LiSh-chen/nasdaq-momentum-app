import { buildContent, extOf, playerHtml, tabsOf, viewerUrl, youtubeWatchUrl } from './build';

const raw = {
  url: 'https://ocw.example.edu/course/101/',
  title: '微積分 - 開放式課程',
  heading: '微積分（一）',
  headings: [{ level: 1, text: '微積分（一）' }, { level: 2, text: '課程大綱' }, { level: 2, text: '影片' }, { level: 2, text: '講義' }],
  paras: ['本課程介紹極限、導數與積分的基本概念與應用。', '本課程介紹極限、導數與積分的基本概念與應用。', '授課教師：王大明', '下載'],
  frames: [
    { src: 'https://www.youtube.com/embed/dQw4w9WgXcQ?rel=0', title: '第一講', section: '影片' },
    { src: 'https://www.youtube-nocookie.com/embed/videoseries?list=PLabc123', title: '', section: '影片' },
    { src: 'https://www.googletagmanager.com/ns.html?id=GTM-1', title: '', section: '' },
  ],
  videos: [{ src: 'https://cdn.example.edu/v/lec2.mp4', title: '第二講', section: '影片' }],
  links: [
    { href: 'https://youtu.be/9bZkp7q19f0', text: '第三講', section: '影片' },
    { href: 'https://www.youtube.com/embed/dQw4w9WgXcQ', text: '重複的第一講', section: '影片' },
    { href: 'https://ocw.example.edu/files/notes-1.pdf', text: '第一章講義', section: '講義' },
    { href: 'https://ocw.example.edu/files/hw1.pdf', text: '', section: '作業' },
    { href: 'https://ocw.example.edu/files/midterm.pdf', text: '期中考', section: '' },
    { href: 'https://ocw.example.edu/files/syllabus.pdf', text: '課程大綱', section: '' },
    { href: 'https://ocw.example.edu/files/notes-1.pdf', text: '第一章講義（重複）', section: '講義' },
    { href: 'https://ocw.example.edu/course/101/week-2/', text: '第二週', section: '' },
    { href: 'https://ocw.example.edu/course/101/', text: '本頁', section: '' },
    { href: 'https://other.org/page', text: '外部', section: '' },
  ],
};

test('擷取影片：YouTube 嵌入、播放清單、mp4、連結，且去除重複與非影片 iframe', () => {
  const c = buildContent(raw);
  expect(c.videos.map((v) => [v.kind, v.ref])).toEqual([
    ['youtube', 'dQw4w9WgXcQ'],
    ['youtube-playlist', 'PLabc123'],
    ['mp4', 'https://cdn.example.edu/v/lec2.mp4'],
    ['youtube', '9bZkp7q19f0'],
  ]);
  expect(c.videos[0].title).toBe('第一講');
  expect(c.videos[1].title).toBe('影片'); // 沒有標題時改用所在區塊
});

test('檔案依內容分類並去重，沒有連結文字時用檔名', () => {
  const c = buildContent(raw);
  const by = Object.fromEntries(c.materials.map((m) => [m.title, m.category]));
  expect(by['第一章講義']).toBe('lecture');
  expect(by['hw1.pdf']).toBe('assignment');
  expect(by['期中考']).toBe('exam');
  expect(by['課程大綱']).toBe('overview');
  expect(c.materials).toHaveLength(4);
});

test('相關頁面只含同網站、非檔案非影片、非本頁的連結', () => {
  const c = buildContent(raw);
  expect(c.related.map((r) => r.title)).toEqual(['第二週']);
});

test('標題、大綱與說明文字整理', () => {
  const c = buildContent(raw);
  expect(c.title).toBe('微積分（一）');
  expect(c.source).toBe('ocw.example.edu');
  expect(c.overview).toEqual(['本課程介紹極限、導數與積分的基本概念與應用。', '授課教師：王大明']); // 去重、保留短資訊、丟掉按鈕文字
  expect(c.outline[1]).toEqual({ level: 2, text: '課程大綱' });
});

test('tabsOf 只列出有內容的分頁，順序固定', () => {
  expect(tabsOf(buildContent(raw)).map((t) => t.key)).toEqual(['overview', 'videos', 'lecture', 'assignment', 'exam', 'related']);
  const empty = buildContent({ url: 'https://a.b/', title: 'x' });
  expect(tabsOf(empty)).toEqual([]);
});

test('播放器網頁：YouTube、播放清單、mp4', () => {
  expect(playerHtml({ kind: 'youtube', ref: 'abcdefghijk' })).toContain('youtube-nocookie.com/embed/abcdefghijk');
  expect(playerHtml({ kind: 'youtube-playlist', ref: 'PL1' })).toContain('videoseries?list=PL1');
  expect(playerHtml({ kind: 'mp4', ref: 'https://x/y.mp4' })).toContain('<video src="https://x/y.mp4"');
  expect(playerHtml({ kind: 'mp4', ref: 'https://x/a"onerror="x.mp4' })).not.toContain('a"onerror');
});

test('輔助函式', () => {
  expect(extOf('https://a/b/c.PDF?x=1')).toBe('pdf');
  expect(youtubeWatchUrl({ kind: 'youtube', ref: 'abcdefghijk' })).toBe('https://www.youtube.com/watch?v=abcdefghijk');
  expect(viewerUrl('https://a/b.pdf?x=1&y=2')).toBe('https://docs.google.com/gview?embedded=1&url=https%3A%2F%2Fa%2Fb.pdf%3Fx%3D1%26y%3D2');
});
