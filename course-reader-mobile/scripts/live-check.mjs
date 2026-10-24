// 對真實的 MIT Learn API 與 OCW 網站驗證 src/api 的解析是否正確（需要網路）
import { searchCourses } from '../src/api/mitLearn.js';
import { fetchOutline, flattenOutline } from '../src/api/ocw.js';

let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); if (!ok) failed++; };

const base = await searchCourses({ q: 'physics', offeredBy: 'ocw' });
check(base.items.length > 0 && base.count > 0, `搜尋 physics/ocw：${base.count} 門，取回 ${base.items.length} 筆`);
check(base.items.every((c) => c.title && c.url?.startsWith('https://')), '每筆都有標題與網址');
check(base.items.every((c) => c.unified), 'OCW 課程皆標為統一閱讀');
check(base.facets.level.length > 0 && base.facets.topic.length > 0 && base.facets.offered_by.length > 0, `篩選統計：level ${base.facets.level.map((f) => f.key)} / topic ${base.facets.topic.length} 項 / source ${base.facets.offered_by.map((f) => f.key)}`);
console.log('  範例:', JSON.stringify({ ...base.items[0], description: base.items[0].description.slice(0, 60) }));

const lv = await searchCourses({ q: 'physics', offeredBy: 'ocw', level: 'graduate' });
check(lv.count > 0 && lv.count < base.count, `level=graduate 縮小結果：${base.count} → ${lv.count}`);
const tp = await searchCourses({ offeredBy: 'ocw', topic: 'Physics' });
check(tp.count > 0, `topic=Physics：${tp.count} 門`);
const vid = await searchCourses({ offeredBy: 'ocw', hasVideos: true });
check(vid.count > 0, `有講課影片：${vid.count} 門`);
const pop = await searchCourses({ offeredBy: 'ocw', sort: '-views' });
const def = await searchCourses({ offeredBy: 'ocw' });
check(pop.items[0].id !== def.items[0].id, `熱門排序改變順序：${pop.items[0].title} vs ${def.items[0].title}`);
const page2 = await searchCourses({ offeredBy: 'ocw', offset: 20 });
check(page2.items[0].id !== def.items[0].id && page2.hasMore, '分頁 offset=20 取得下一批');
const mitx = await searchCourses({ offeredBy: 'mitx' });
check(mitx.items.length > 0 && mitx.items.every((c) => !c.unified), `MITx：${mitx.count} 門，皆為網頁開啟`);

// 搜尋結果裡常是頁面很少的短期課程；另以一門內容豐富的已知課程驗證章節解析
const SMALL = (await searchCourses({ q: 'weather climate', offeredBy: 'ocw' })).items[0];
const small = flattenOutline(await fetchOutline(SMALL.url));
// 較新的課程首頁沒有章節連結（解析為 0 頁是正常的），App 會退回直接開啟課程首頁
console.log(`INFO  搜尋結果中的課程「${SMALL.title}」解析出 ${small.length} 頁 ${SMALL.url}`);
if (small.length === 0) {
  const r = await fetch(SMALL.url);
  const h = await r.text();
  console.log(`  [debug] status=${r.status} final=${r.url} len=${h.length}`);
  console.log('  [debug] hrefs:', [...new Set([...h.matchAll(/href="([^"]*)"/g)].map((m) => m[1]).filter((x) => /course|pages|resources/.test(x)))].slice(0, 15));
}
const picked = { c: { title: '8.01SC Classical Mechanics', url: 'https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/' } };
picked.o = await fetchOutline(picked.c.url);
picked.pages = flattenOutline(picked.o);
const outline = picked.o;
const pages = picked.pages;
console.log('  測試課程:', picked.c.title, picked.c.url);
check(outline.length >= 3 && pages.length >= 10, `章節 ${outline.length} 個、頁面 ${pages.length} 個`);
console.log('  前幾頁:', pages.slice(0, 6).map((p) => `${p.section} > ${p.title}`));

let hasMain = 0;
const sample = [pages[0], pages[Math.floor(pages.length / 2)], pages[pages.length - 1]];
for (const p of sample) {
  const res = await fetch(p.url);
  const html = await res.text();
  const main = html.includes('id="course-main-content"');
  if (main) hasMain++;
  console.log(`  頁面 ${res.status} main=${main} video=${(html.match(/<video/g) || []).length} yt=${(html.match(/youtube\.com\/embed|youtube-nocookie/g) || []).length} archive=${(html.match(/archive\.org/g) || []).length} ${p.url}`);
}
check(hasMain === sample.length, `抽樣頁面皆含 #course-main-content（${hasMain}/${sample.length}）`);

// 影片列表頁
const vidPages = pages.filter((p) => /video_galleries/.test(p.url));
if (vidPages[0]) {
  const html = await (await fetch(vidPages[0].url)).text();
  console.log(`  影片頁 main=${html.includes('course-main-content')} youtube-embed=${(html.match(/youtube\.com\/embed/g) || []).length} mp4=${(html.match(/\.mp4/g) || []).length} ${vidPages[0].url}`);
}
// 影片資源頁：看實際的播放器標記，判斷 App 內能否直接播放
const GAL = 'https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/video_galleries/lecture-videos/';
{
  const gal = await (await fetch(GAL)).text();
  const res = [...new Set([...gal.matchAll(/href="([^"]*\/resources\/[^"#?]+)"/g)].map((m) => new URL(m[1], GAL).toString()))];
  console.log(`  影片列表頁: 資源連結 ${res.length} 個, main=${gal.includes('course-main-content')}`);
  if (res[0]) {
    const h = await (await fetch(res[0])).text();
    const n = (re) => (h.match(re) || []).length;
    console.log(`  影片資源頁 ${res[0]}`);
    console.log(`   <video>=${n(/<video/g)} <iframe>=${n(/<iframe/g)} youtube=${n(/youtube/gi)} archive.org=${n(/archive\.org/g)} .mp4=${n(/\.mp4/g)} .vtt=${n(/\.vtt/g)} videojs=${n(/video-js|videojs/gi)} main=${h.includes('course-main-content')}`);
    console.log('   iframe src:', [...h.matchAll(/<iframe[^>]*src="([^"]+)"/g)].map((m) => m[1]).slice(0, 3));
    console.log('   youtube ids:', [...new Set([...h.matchAll(/(?:youtube(?:-nocookie)?\.com\/(?:embed\/|watch\?v=)|youtu\.be\/)([\w-]{11})/g)].map((m) => m[1]))].slice(0, 3));
    console.log('   mp4:', [...new Set([...h.matchAll(/https?:[^"' ]+\.mp4/g)].map((m) => m[0]))].slice(0, 2));
  }
}
const resLinks = [];
for (const p of pages.filter((p) => /video|lecture/i.test(p.url + p.title)).slice(0, 4)) {
  const html = await (await fetch(p.url)).text();
  const found = [...html.matchAll(/href="([^"]*\/resources\/[^"#?]+)"/g)].map((m) => new URL(m[1], p.url).toString());
  console.log(`  ${p.title}: 內含資源連結 ${new Set(found).size} 個, <iframe>=${(html.match(/<iframe/g) || []).length} <video>=${(html.match(/<video/g) || []).length}`);
  resLinks.push(...found);
  if (resLinks.length) break;
}
const first = [...new Set(resLinks)][0];
if (first) {
  const html = await (await fetch(first)).text();
  const pat = (re) => (html.match(re) || []).length;
  console.log(`  資源頁 ${first}`);
  console.log(`   <video>=${pat(/<video/g)} <iframe>=${pat(/<iframe/g)} youtube=${pat(/youtube/gi)} archive.org=${pat(/archive\.org/g)} .mp4=${pat(/\.mp4/g)} .pdf=${pat(/\.pdf/g)} video.js=${pat(/video-js|videojs/gi)} main=${html.includes('course-main-content')}`);
  const ifr = [...html.matchAll(/<iframe[^>]*src="([^"]+)"/g)].map((m) => m[1]).slice(0, 3);
  const yt = [...html.matchAll(/(?:youtube\.com\/(?:embed\/|watch\?v=)|youtu\.be\/)([\w-]{11})/g)].map((m) => m[1]).slice(0, 3);
  console.log('   iframe src:', ifr, ' youtube ids:', yt);
}
console.log(failed ? `\n${failed} 項失敗` : '\n全部通過');
process.exit(failed ? 1 : 0);
