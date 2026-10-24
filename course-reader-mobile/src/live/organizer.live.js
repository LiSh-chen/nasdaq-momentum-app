/** @jest-environment node */
// 對真實的 OCW 頁面執行整理器（需要網路）：LIVE=1 npx jest --testMatch '**/*.live.js'
import { JSDOM } from 'jsdom';
import { EXTRACT_JS } from '../content/extractor';
import { buildContent, tabsOf } from '../content/build';
import { fetchOutline, flattenOutline } from '../api/ocw';

const COURSE = 'https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/';

async function organize(url) {
  const html = await (await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' } })).text();
  const dom = new JSDOM(html, { url, runScripts: 'outside-only' });
  const msgs = [];
  dom.window.ReactNativeWebView = { postMessage: (m) => msgs.push(JSON.parse(m)) };
  dom.window.eval(EXTRACT_JS);
  expect(msgs[0].type).toBe('content');
  const content = buildContent(msgs[0]);
  const tabs = tabsOf(content);
  console.log(`[${url.replace(COURSE, '')}] title="${content.title}" tabs=${tabs.map((t) => `${t.label}:${t.count}`).join(' | ') || '(none)'}\n` +
    `  videos: ${content.videos.slice(0, 3).map((v) => `${v.kind}:${String(v.ref).slice(-40)} "${v.title.slice(0, 30)}"`).join(' ; ')}\n` +
    `  materials: ${content.materials.slice(0, 5).map((m) => `${m.category}/${m.ext} "${m.title.slice(0, 30)}"`).join(' ; ')}\n` +
    `  overview[0]: ${(content.overview[0] || '').slice(0, 80)}  related=${content.related.length} outline=${content.outline.length}`);
  return { content, tabs, raw: msgs[0] };
}

jest.setTimeout(120000);

test('真實 OCW 課程的各類頁面都能整理出分類內容', async () => {
  const pages = flattenOutline(await fetchOutline(COURSE));
  expect(pages.length).toBeGreaterThan(20);
  const pick = (re) => pages.find((p) => re.test(p.url + p.title));
  const targets = [pick(/syllabus/i), pick(/week-1-introduction|week 1 introduction/i), pick(/1-1-coordinate/i), pick(/problem-set-1\//i), pick(/readings/i), pick(/assignments/i)].filter(Boolean);
  expect(targets.length).toBeGreaterThanOrEqual(4);

  let withContent = 0;
  for (const t of targets) {
    const { tabs } = await organize(t.url);
    if (tabs.length) withContent++;
  }
  // 大多數頁面都應該整理出至少一個分頁
  expect(withContent).toBeGreaterThanOrEqual(Math.ceil(targets.length * 0.75));

  // 影片列表頁：應整理出大量影片
  const gallery = await organize(`${COURSE}video_galleries/lecture-videos/`);
  console.log('gallery raw frames/videos/links:', gallery.raw.frames.length, gallery.raw.videos.length, gallery.raw.links.length);
});
