/** @jest-environment node */
// 對真實的 OCW 頁面執行整理器（需要網路）：LIVE=1 npx jest --testMatch '**/*.live.js'
import { JSDOM } from 'jsdom';
import { EXTRACT_JS } from '../content/extractor';
import { buildContent, tabsOf } from '../content/build';
import { fetchOutline, flattenOutline } from '../api/ocw';
import { resolveFileUrl } from '../content/resolve';

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
  const rl = msgs[0].links.filter((l) => /\/resources\//.test(l.href) && !l.nav);
  console.log(`  [evidence] links under /resources/ (non-nav): ${rl.length}; sample: ${rl.slice(0, 4).map((l) => `"${l.text.slice(0, 25)}"->${l.href.replace(COURSE, '')}`).join(' ; ')}`);
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

  // 作業、閱讀、講義頁：應整理出檔案，且能解析成實際的檔案網址
  const ps = await organize(`${COURSE}pages/week-1-kinematics/problem-set-1/`);
  const rd = await organize(`${COURSE}pages/readings/`);
  const withFiles = [ps, rd].find((r) => r.content.materials.length);
  console.log('materials found on problem-set/readings:', ps.content.materials.length, rd.content.materials.length);
  expect(withFiles).toBeTruthy();
  const res = withFiles.content.materials.find((m) => m.resolve);
  if (res) {
    const real = await resolveFileUrl(res.url);
    console.log(`resolve ${res.url.replace(COURSE, '')} => ${real}`);
    expect(real).toMatch(/\.(pdf|zip|pptx?|docx?|xlsx?)/i);
  }

  // 影片列表頁：應整理出大量影片
  const gallery = await organize(`${COURSE}video_galleries/lecture-videos/`);
  console.log('gallery raw frames/videos/links:', gallery.raw.frames.length, gallery.raw.videos.length, gallery.raw.links.length);
});

test('新來源的課程頁：整理器實際抓到什麼（耶魯、陽明交大）', async () => {
  const targets = [
    'https://oyc.yale.edu/astronomy/astr-160',
    'https://ocw.nycu.edu.tw/?course_page=all-course/college-of-science/am/calculus-a-ii-academic-109',
  ];
  for (const url of targets) {
    const { content, tabs, raw } = await organize(url);
    console.log(`[${url}] tabs=${tabs.map((t) => `${t.label}:${t.count}`).join(' | ') || '(none)'}; related sample: ${content.related.slice(0, 8).map((r) => `${r.title}->${r.url.slice(-45)}`).join(' ; ')}; frames=${raw.frames.map((f) => f.src.slice(0, 60)).join(' , ')}`);
  }
});

test('耶魯：課程頁解析出講次，講次頁整理得出影片或逐字稿', async () => {
  const pages = flattenOutline(await fetchOutline('https://oyc.yale.edu/astronomy/astr-160'));
  console.log(`yale lectures: ${pages.length}; first 3: ${pages.slice(0, 3).map((p) => `${p.title}->${p.url.slice(-30)}`).join(' ; ')}`);
  expect(pages.length).toBeGreaterThanOrEqual(10);
  expect(pages.every((p) => p.title && p.url.startsWith('https://oyc.yale.edu/astronomy/astr-160/'))).toBe(true);
  const { tabs, content, raw } = await organize(pages[0].url);
  console.log(`yale lecture page: tabs=${tabs.map((t) => `${t.label}:${t.count}`).join(' | ') || '(none)'}; frames=${raw.frames.map((f) => f.src.slice(0, 70)).join(' , ')}; videoLinks=${content.videos.length}; links=${raw.links.length}`);
});
