/** @jest-environment node */
import { JSDOM } from 'jsdom';
import { EXTRACT_JS } from './extractor';
import { buildContent, tabsOf } from './build';

// 在模擬的網頁環境中實際執行注入腳本，取得它回傳的訊息
function extract(html, url = 'https://ocw.example.edu/course/101/') {
  const dom = new JSDOM(html, { url, runScripts: 'outside-only' });
  const messages = [];
  dom.window.ReactNativeWebView = { postMessage: (m) => messages.push(JSON.parse(m)) };
  dom.window.eval(EXTRACT_JS);
  return messages[0];
}

const PAGE = `<!doctype html><html lang="zh-TW"><head><title>微積分 | 開放式課程</title></head><body>
<nav><a href="/">首頁</a><a href="/courses">課程列表</a></nav>
<main>
  <h1>微積分（一）</h1>
  <h2>課程大綱</h2>
  <p>本課程介紹極限、導數與積分的基本概念，並練習以微積分解決實際問題。</p>
  <a href="/files/syllabus.pdf">課程大綱 PDF</a>
  <h2>課程影片</h2>
  <div class="vid"><iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="第一講"></iframe></div>
  <div class="vid"><iframe data-src="https://www.youtube.com/embed/9bZkp7q19f0"></iframe></div>
  <h2>講義</h2>
  <ul><li><a href="/files/ch1.pdf">第一章講義</a></li><li><a href="files/ch2.pptx">第二章投影片</a></li></ul>
  <h2>作業</h2>
  <ul><li><a href="/files/hw1.pdf">作業一</a></li></ul>
  <a href="mailto:ta@example.edu">助教</a>
</main></body></html>`;

test('擷取影片、檔案連結與大綱，並標註所在區塊', () => {
  const raw = extract(PAGE);
  expect(raw.type).toBe('content');
  expect(raw.heading).toBe('微積分（一）');
  expect(raw.lang).toBe('zh-TW');
  expect(raw.frames.map((f) => f.src)).toEqual([
    'https://www.youtube.com/embed/dQw4w9WgXcQ',
    'https://www.youtube.com/embed/9bZkp7q19f0', // data-src 也會讀到
  ]);
  expect(raw.frames[0].section).toBe('課程影片');
  const hw = raw.links.find((l) => l.href.endsWith('hw1.pdf'));
  expect(hw.section).toBe('作業');
  expect(raw.links.find((l) => l.href.endsWith('ch2.pptx')).href).toBe('https://ocw.example.edu/course/101/files/ch2.pptx'); // 相對路徑轉絕對
  expect(raw.links.some((l) => l.href.startsWith('mailto:'))).toBe(false);
  expect(raw.headings.map((h) => h.text)).toEqual(['微積分（一）', '課程大綱', '課程影片', '講義', '作業']);
});

test('擷取結果交給 buildContent 後得到正確的分類', () => {
  const content = buildContent(extract(PAGE));
  expect(content.videos).toHaveLength(2);
  const cat = Object.fromEntries(content.materials.map((m) => [m.title, m.category]));
  expect(cat).toMatchObject({ '課程大綱 PDF': 'overview', '第一章講義': 'lecture', '第二章投影片': 'lecture', '作業一': 'assignment' });
  expect(tabsOf(content).map((t) => t.key)).toEqual(['overview', 'videos', 'lecture', 'assignment']); // <nav> 內的連結不算相關頁面
});

test('遇到 Cloudflare 驗證頁時回報 challenge，不當成內容', () => {
  const raw = extract('<html><head><title>Just a moment...</title></head><body><p>Checking</p></body></html>');
  expect(raw).toEqual({ type: 'challenge' });
});

test('頁面沒有 main 時退回整個 body，不會出錯', () => {
  const raw = extract('<html><head><title>t</title></head><body><h2>影片</h2><iframe src="https://youtu.be/dQw4w9WgXcQ"></iframe></body></html>');
  expect(raw.type).toBe('content');
  expect(raw.frames).toHaveLength(1);
});

test('排除選單與側欄：說明文字不含選單項目，選單連結標記為 nav', () => {
  const raw = extract(`<html><head><title>t</title></head><body><main>
    <div class="course-nav"><ul><li>Syllabus About the Team Readings</li><li><a href="/pages/week-1/">Week 1</a></li></ul></div>
    <h1>Classical Mechanics</h1>
    <p>This course introduces classical mechanics for first-year students.</p>
    <a href="/pages/extra/">延伸閱讀</a>
  </main></body></html>`);
  expect(raw.paras).toEqual(['This course introduces classical mechanics for first-year students.']);
  expect(raw.links.find((l) => l.href.endsWith('/week-1/')).nav).toBe(true);
  expect(raw.links.find((l) => l.href.endsWith('/extra/')).nav).toBe(false);
});

describe('版面判斷不能誤傷內容', () => {
  test('內容外層的容器 class 帶有 nav／menu 字樣時，內容仍要保留', () => {
    const raw = extract(`<html><head><title>t</title></head><body class="has-navigation menu-open">
      <div class="layout-container with-sidebar-menu"><main>
        <h1>Astronomy</h1><h2>Course Description</h2>
        <p>This course covers the frontiers of modern astrophysics research.</p>
        <a href="/astronomy/astr-160/lecture-1">Lecture 1</a>
      </main></div></body></html>`);
    expect(raw.headings.map((h) => h.text)).toEqual(['Astronomy', 'Course Description']);
    expect(raw.paras).toHaveLength(1);
    expect(raw.links.find((l) => l.href.endsWith('lecture-1')).nav).toBe(false);
  });

  test('文章內的 <header> 不是版面，裡面的標題要保留', () => {
    const raw = extract('<html><head><title>t</title></head><body><article><header><h1>課程名稱</h1></header><p>本課程介紹基本概念與應用方式。</p></article></body></html>');
    expect(raw.headings.map((h) => h.text)).toEqual(['課程名稱']);
  });

  test('真正的選單（nav 標籤、role、class 標記）仍被排除', () => {
    const raw = extract(`<html><head><title>t</title></head><body><main>
      <ul class="main-menu"><li><a href="/a">A</a></li></ul>
      <div role="navigation"><a href="/b">B</a></div>
      <div class="course-nav"><a href="/c">C</a></div>
      <div class="navbar-top"><a href="/e">E</a></div>
      <aside><a href="/d">D</a></aside>
      <a href="/ok">內容連結</a></main></body></html>`);
    const nav = Object.fromEntries(raw.links.map((l) => [l.href.split('/').pop(), l.nav]));
    expect(nav).toEqual({ a: true, b: true, c: true, e: true, d: true, ok: false });
  });
});

test('延遲載入的 iframe：src 是佔位圖時改用 data-src，並略過非網頁位址', () => {
  const raw = extract(`<html><head><title>t</title></head><body><main>
    <iframe src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" data-src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="第一講"></iframe>
    <iframe src="about:blank" data-lazy-src="https://player.vimeo.com/video/76979871"></iframe>
    <iframe src="data:text/html;base64,AAAA"></iframe>
  </main></body></html>`);
  expect(raw.frames.map((f) => f.src)).toEqual(['https://www.youtube.com/embed/dQw4w9WgXcQ', 'https://player.vimeo.com/video/76979871']);
});
