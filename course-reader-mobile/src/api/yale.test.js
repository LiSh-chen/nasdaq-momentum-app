import { parseYaleListing, resetYaleCache, searchYale, toCourse } from './yale';

// 欄位與 class 名稱取自 oyc.yale.edu/courses 的實際表格
const row = (dept, deptHref, code, href, title, prof, sem) => `
<tr>
<td headers="view-title-table-column" class="views-field views-field-title is-active views-align-left"><a href="${deptHref}" hreflang="en">${dept}</a> </td>
<td headers="view-field-course-number-table-column" class="views-field views-field-field-course-number views-align-left"><a href="${href}" hreflang="en">${code}</a> </td>
<td headers="view-title-1-table-column" class="views-field views-field-title-1 views-align-left"><a href="${href}" hreflang="en">${title}</a> </td>
<td headers="view-field-course-professor-last-name-table-column" class="views-field views-field-field-course-professor-last-name views-align-left">${prof} </td>
<td headers="view-field-semester-table-column" class="views-field views-field-field-semester views-align-left">${sem} </td>
</tr>`;

const HTML = `<table><thead><tr><th>Subject</th></tr></thead><tbody>
${row('American Studies', '/american-studies', 'AMST 246', '/american-studies/amst-246', 'Hemingway, Fitzgerald, Faulkner', 'Wai Chee Dimock', 'Fall 2011')}
${row('Astronomy', '/astronomy', 'ASTR 160', '/astronomy/astr-160', 'Frontiers and Controversies in Astrophysics', 'Charles Bailyn', 'Spring 2007')}
${row('Astronomy', '/astronomy', 'ASTR 160', '/astronomy/astr-160', 'Frontiers and Controversies in Astrophysics', 'Charles Bailyn', 'Spring 2007')}
</tbody></table>`;

const ok = (html) => jest.fn().mockResolvedValue({ ok: true, text: async () => html });
beforeEach(resetYaleCache);

test('解析課程列表：名稱、系所、課號、教授、學期，並去除重複', () => {
  const list = parseYaleListing(HTML);
  expect(list).toHaveLength(2);
  expect(list[1]).toEqual({
    title: 'Frontiers and Controversies in Astrophysics', url: 'https://oyc.yale.edu/astronomy/astr-160',
    dept: 'Astronomy', code: 'ASTR 160', professor: 'Charles Bailyn', semester: 'Spring 2007',
  });
});

test('轉成統一的課程格式（免登入、有講課影片）', () => {
  const c = toCourse(parseYaleListing(HTML)[1]);
  expect(c).toMatchObject({
    id: 'yale:astronomy/astr-160', sourceCode: 'yale', sourceName: 'Open Yale Courses', free: true, requiresLogin: false, unified: true,
    instructors: ['Charles Bailyn'], topics: ['Astronomy'], features: ['Lecture Videos'],
  });
  expect(c.description).toContain('ASTR 160');
});

test('搜尋依名稱、系所、教授本機篩選；空白關鍵字回傳全部', async () => {
  const f = ok(HTML);
  expect((await searchYale('astro', f)).items.map((i) => i.title)).toEqual(['Frontiers and Controversies in Astrophysics']);
  expect((await searchYale('FAULKNER', f)).items).toHaveLength(1);
  expect((await searchYale('dimock', f)).items[0].title).toMatch(/Hemingway/);
  expect((await searchYale('', f)).count).toBe(2);
  expect((await searchYale('zzz', f)).items).toEqual([]);
  expect(f).toHaveBeenCalledTimes(1); // 列表只讀一次並暫存
});

test('讀取失敗時丟出錯誤', async () => {
  await expect(searchYale('a', jest.fn().mockResolvedValue({ ok: false, status: 503 }))).rejects.toThrow('503');
});
