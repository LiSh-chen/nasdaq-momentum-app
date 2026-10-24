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
  expect(by['hw1']).toBe('assignment'); // 沒有連結文字時用檔名（去掉副檔名）
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

describe('真實 OCW 頁面暴露的情況', () => {
  const page = (extra) => ({ url: 'https://ocw.example.edu/courses/c1/pages/lesson/', title: 't', ...extra });

  test('有嵌入影片時，「下載影片」連結不重複列出', () => {
    const c = buildContent(page({
      frames: [{ src: 'https://www.youtube.com/embed/abcdefghijk', title: '', section: 'Lesson 1' }],
      links: [{ href: 'https://archive.org/download/X/lesson_360p.mp4', text: 'download the video', section: 'Lesson 1' }],
    }));
    expect(c.videos).toHaveLength(1);
    expect(c.videos[0]).toMatchObject({ kind: 'youtube', title: 'Lesson 1' });
  });

  test('只有 mp4 連結時仍列為影片，標題無意義時改用檔名', () => {
    const c = buildContent(page({
      links: [
        { href: 'https://archive.org/download/X/MIT8_01F16_Demo_02_360p.mp4', text: 'video 5 MB', section: '' },
        { href: 'https://archive.org/download/X/Lecture_03.mp4', text: '第三講', section: '' },
      ],
    }));
    expect(c.videos.map((v) => v.title)).toEqual(['MIT8 01F16 Demo 02', '第三講']);
  });

  test('OCW 資源頁連結視為檔案：靠「(PDF)」等提示判斷，標題去掉提示，點開時才解析實際網址', () => {
    const c = buildContent(page({
      links: [
        { href: 'https://ocw.example.edu/courses/c1/resources/mit8_01f16_pset1_new/', text: 'Problem Set 1 (PDF)', section: 'Assignments' },
        { href: 'https://ocw.example.edu/courses/c1/resources/lec3_transcript_pdf/', text: 'English', section: 'Lecture 3' },
        { href: 'https://ocw.example.edu/courses/c1/resources/solutions-1/', text: 'Solutions', section: 'Problem Set 1 Solutions' },
      ],
    }));
    expect(c.materials).toMatchObject([
      { title: 'Problem Set 1', ext: 'pdf', resolve: true, category: 'assignment' },
      { title: 'lec3 transcript', ext: 'pdf', resolve: true, category: 'lecture' }, // 網址結尾 _pdf 也算提示；逐字稿歸講義
      { title: 'Solutions', ext: 'file', resolve: true, category: 'assignment' }, // 沒提示但能從區塊歸類
    ]);
    expect(c.related).toEqual([]); // 資源頁不算相關頁面
  });

  test('資源頁連結中的翻頁、選單與影片頁不算檔案', () => {
    const c = buildContent(page({
      links: [
        { href: 'https://ocw.example.edu/courses/c1/resources/ps-1-4/', text: 'Previous', section: '' },
        { href: 'https://ocw.example.edu/courses/c1/resources/ps-1-6/', text: 'Next »', section: '' },
        { href: 'https://ocw.example.edu/courses/c1/resources/lesson-1/', text: 'Lesson 1', section: 'Lecture Videos' },
        { href: 'https://ocw.example.edu/courses/c1/resources/x/', text: 'Menu item', section: '', nav: true },
        { href: 'https://ocw.example.edu/courses/c1/resources/misc/', text: '其他東西', section: '' },
      ],
    }));
    expect(c.materials).toEqual([]);
  });

  test('說明文字排除翻頁列', () => {
    const c = buildContent(page({ paras: ['« Previous | Next »', 'Next lesson is available soon', '本課程介紹基本概念與應用方式。'] }));
    expect(c.overview).toEqual(['本課程介紹基本概念與應用方式。']); // 「Next lesson…」開頭也被視為翻頁文字
  });

  test('選單連結不列為相關頁面；有章節時不顯示相關頁面分頁', () => {
    const c = buildContent(page({
      links: [
        { href: 'https://ocw.example.edu/courses/c1/pages/week-2/', text: 'Week 2', section: '', nav: true },
        { href: 'https://ocw.example.edu/courses/c1/pages/extra/', text: '延伸閱讀', section: '', nav: false },
      ],
    }));
    expect(c.related.map((r) => r.title)).toEqual(['延伸閱讀']);
    expect(tabsOf(c).map((t) => t.key)).toContain('related');
    expect(tabsOf(c, { hideRelated: true }).map((t) => t.key)).not.toContain('related');
  });
});

test('檔名是雜湊時改用連結文字或所在區塊當標題', () => {
  const c = buildContent({
    url: 'https://ocw.example.edu/c/', title: 't',
    links: [
      { href: 'https://ocw.example.edu/c/637ca1c7a1cac1f493c781ec6a3f54_lecture.pdf', text: 'English', section: '' },
      { href: 'https://ocw.example.edu/c/637ca1c7a1cac1f493c781ec6a3f54a1.pdf', text: 'English', section: 'Lecture Transcript' },
      { href: 'https://ocw.example.edu/c/637ca1c7a1cac1f493c781ec6a3f54a2.pdf', text: '', section: '' },
    ],
  });
  expect(c.materials.map((m) => m.title)).toEqual(['lecture', 'Lecture Transcript（English）', 'PDF 檔案']);
});

describe('陽明交大等真實頁面暴露的情況', () => {
  const page = (extra) => ({ url: 'https://ocw.nycu.edu.tw/?course_page=all-course/x/y/z', title: 't', ...extra });

  test('標題去掉站名後綴', () => {
    expect(buildContent(page({ title: '微積分甲(二) | 應用數學系 蘇承芳老師 - NYCU OCW' })).title).toBe('微積分甲(二) | 應用數學系 蘇承芳老師');
    expect(buildContent(page({ title: 'Syllabus | MIT OpenCourseWare' })).title).toBe('Syllabus');
    expect(buildContent(page({ title: '一個沒有後綴的標題' })).title).toBe('一個沒有後綴的標題');
    expect(buildContent(page({ title: 'Intro - Part 2 of the long running lecture series on physics' })).title).toBe('Intro - Part 2 of the long running lecture series on physics');
    expect(buildContent(page({ heading: '微積分', title: '微積分 - NYCU OCW' })).title).toBe('微積分'); // 有 h1 就用 h1
  });

  test('說明文字排除 CSS／程式碼片段', () => {
    const c = buildContent(page({ paras: ['.。 {<br /> font-size: 18px;<br /> }', 'body { margin: 0; padding: 0 }', '本課程介紹微積分的基本概念。'] }));
    expect(c.overview).toEqual(['本課程介紹微積分的基本概念。']);
  });

  test('相關頁面的連結文字太籠統（如「線上觀看」）時，用所在區塊的標題區分', () => {
    const c = buildContent(page({
      links: [
        { href: 'https://ocw.nycu.edu.tw/?post_type=course_page&p=1', text: '線上觀看', section: '微積分', context: '第一週 極限' },
        { href: 'https://ocw.nycu.edu.tw/?post_type=course_page&p=2', text: '線上觀看', section: '微積分', context: '第二週 導數' },
        { href: 'https://ocw.nycu.edu.tw/?post_type=course_page&p=3', text: '線上觀看', section: '微積分' },
      ],
    }));
    // 優先用同一列的文字（哪一週），其次用所在區塊標題
    expect(c.related.map((r) => r.title)).toEqual(['第一週 極限・線上觀看', '第二週 導數・線上觀看', '微積分・線上觀看']);
  });
});

test('檔名就是頁面上某部影片的代碼時，視為該影片的逐字稿', () => {
  const c = buildContent({
    url: 'https://ocw.example.edu/c/p/', title: 't',
    frames: [{ src: 'https://www.youtube.com/embed/F3N5EkMX_ks', title: 'Course Introduction', section: '' }],
    links: [
      { href: 'https://ocw.example.edu/c/F3N5EkMX_ks.pdf', text: 'English', section: '' },
      { href: 'https://ocw.example.edu/c/notes.pdf', text: '講義', section: '' },
    ],
  });
  expect(c.materials.map((m) => [m.title, m.category])).toEqual([['Course Introduction（逐字稿）', 'lecture'], ['講義', 'lecture']]);
});

test('嵌入影片自帶的通用標題（YouTube video player）改用頁面標題', () => {
  const one = buildContent({ url: 'https://oyc.yale.edu/a/b/lecture-1', title: 'ASTR 160 - Yale', heading: 'Lecture 1: Introduction', frames: [{ src: 'https://www.youtube.com/embed/nzmFc2gjUo4', title: 'YouTube video player', section: '' }] });
  expect(one.videos[0].title).toBe('Lecture 1: Introduction');
  const two = buildContent({ url: 'https://x.edu/c', title: '課程', frames: [
    { src: 'https://www.youtube.com/embed/aaaaaaaaaaa', title: 'YouTube video player', section: '' },
    { src: 'https://www.youtube.com/embed/bbbbbbbbbbb', title: 'Video player', section: '' },
    { src: 'https://www.youtube.com/embed/ccccccccccc', title: '第三講', section: '' },
  ] });
  expect(two.videos.map((v) => v.title)).toEqual(['課程 1', '課程 2', '第三講']);
});
