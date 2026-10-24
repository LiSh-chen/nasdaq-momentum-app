import { buildSearchUrl, normalizeCourse, searchCourses } from './mitLearn';

// 欄位結構取自 MIT Learn API 的實際回應
const apiCourse = {
  id: 3308,
  title: 'Infrastructure and Energy Technology Challenges',
  description: '<p>This seminar examines <b>efforts</b>…</p>',
  url: 'https://ocw.mit.edu/courses/11-165-infrastructure-and-energy-technology-challenges-fall-2011/',
  free: true,
  image: { url: 'https://ocw.mit.edu/x.jpg' },
  offered_by: { code: 'ocw', name: 'MIT OpenCourseWare' },
  topics: [{ name: 'Energy' }, { name: 'Economics' }],
  course_feature: ['Lecture Notes'],
  runs: [{ level: [{ name: 'Undergraduate' }, { name: 'Graduate' }], instructors: [{ full_name: 'Prof. Karen R. Polenske' }] }],
};

test('buildSearchUrl 帶入搜尋與篩選參數', () => {
  const u = new URL(buildSearchUrl({ q: ' physics ', offeredBy: 'ocw', level: 'undergraduate', topic: 'Physics', hasVideos: true, sort: '-views', offset: 40 }));
  const p = u.searchParams;
  expect(u.pathname).toBe('/api/v1/learning_resources_search/');
  expect(p.get('resource_type')).toBe('course');
  expect(p.get('q')).toBe('physics');
  expect(p.get('offered_by')).toBe('ocw');
  expect(p.get('level')).toBe('undergraduate');
  expect(p.get('topic')).toBe('Physics');
  expect(p.get('course_feature')).toBe('Lecture Videos');
  expect(p.get('sortby')).toBe('-views');
  expect(p.get('offset')).toBe('40');
  expect(p.getAll('aggregations')).toEqual(['offered_by', 'level', 'topic']);
});

test('buildSearchUrl 未設定的條件不會出現在網址', () => {
  const p = new URL(buildSearchUrl({})).searchParams;
  for (const k of ['q', 'offered_by', 'level', 'topic', 'course_feature', 'sortby']) expect(p.has(k)).toBe(false);
});

test('normalizeCourse 整理欄位，OCW 才使用統一閱讀', () => {
  const c = normalizeCourse(apiCourse);
  expect(c).toMatchObject({
    id: 3308, image: 'https://ocw.mit.edu/x.jpg', sourceCode: 'ocw', unified: true,
    levels: ['Undergraduate', 'Graduate'], topics: ['Energy', 'Economics'], instructors: ['Prof. Karen R. Polenske'],
  });
  expect(c.description).toBe('This seminar examines efforts …');
  expect(normalizeCourse({ ...apiCourse, offered_by: { code: 'mitx', name: 'MITx' } }).unified).toBe(false);
});

test('normalizeCourse 容忍缺少的欄位', () => {
  const c = normalizeCourse({ id: 1, title: 'T', url: 'https://x' });
  expect(c).toMatchObject({ image: null, levels: [], topics: [], instructors: [], unified: false, free: true });
});

test('searchCourses 回傳課程、總數與篩選統計', async () => {
  const fetchImpl = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      count: 90, next: 'https://next', results: [apiCourse],
      metadata: { aggregations: { level: [{ key: 'undergraduate', doc_count: 51 }], topic: [{ key: 'Physics', doc_count: 64 }], offered_by: [{ key: 'ocw', doc_count: 90 }] } },
    }),
  });
  const r = await searchCourses({ q: 'physics' }, fetchImpl);
  expect(r.count).toBe(90);
  expect(r.hasMore).toBe(true);
  expect(r.items[0].title).toBe(apiCourse.title);
  expect(r.facets.level).toEqual([{ key: 'undergraduate', count: 51 }]);
  expect(r.facets.topic[0]).toEqual({ key: 'Physics', count: 64 });
});

test('searchCourses 在 API 失敗時丟出錯誤', async () => {
  await expect(searchCourses({}, jest.fn().mockResolvedValue({ ok: false, status: 500 }))).rejects.toThrow('500');
});
