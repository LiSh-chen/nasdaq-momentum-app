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

const ocw = await searchCourses({ q: 'classical mechanics 8.01', offeredBy: 'ocw' });
const target = ocw.items.find((c) => c.url.includes('8-01sc')) ?? ocw.items[0];
console.log('  測試課程:', target.title, target.url);
const outline = await fetchOutline(target.url);
const pages = flattenOutline(outline);
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
console.log(failed ? `\n${failed} 項失敗` : '\n全部通過');
process.exit(failed ? 1 : 0);
