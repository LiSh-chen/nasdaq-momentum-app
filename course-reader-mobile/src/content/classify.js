// 依標題、檔名與所在區塊標題，把資料歸類到統一的分類
export const CATEGORY_LABELS = {
  overview: '大綱',
  videos: '影片',
  lecture: '講義',
  assignment: '作業',
  exam: '考試',
  reading: '閱讀',
  other: '其他',
};

// 順序即優先權：例如 "exam solutions" 要歸考試，而不是作業
const RULES = [
  ['exam', /考試|期中|期末|測驗|試題|exam|quiz|midterm|final/i],
  ['assignment', /作業|習題|assignment|homework|problem[\s_-]?sets?|\bhw\s?\d|\bps\s?\d|exercises?|projects?/i],
  ['lecture', /講義|筆記|投影片|簡報|教材|lectures?|slides?|notes?|ppt/i],
  ['reading', /閱讀|參考|教科書|課本|textbook|reading|reference/i],
  ['overview', /大綱|綱要|課程簡介|syllabus|schedule|calendar|課表|進度/i],
];

const match = (s) => RULES.find(([, re]) => re.test(s))?.[0] ?? null;

export function classifyMaterial({ title = '', section = '', url = '' }) {
  const file = decodeURIComponent(url.split('?')[0].split('/').pop() || '');
  // 先看檔案自己的名稱，再看它所在區塊的標題
  return match(`${title} ${file}`) ?? match(section) ?? 'other';
}
