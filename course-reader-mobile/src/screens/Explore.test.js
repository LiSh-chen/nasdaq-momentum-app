import React from 'react';
import { FlatList, Share } from 'react-native';
import { render, fireEvent, act } from '@testing-library/react-native';
import Explore from './Explore';
import { searchCatalog } from '../api/search';

jest.mock('../api/search', () => ({ ...jest.requireActual('../api/search'), searchCatalog: jest.fn() }));

const course = (id, title, extra = {}) => ({
  id, title, description: '', image: null, url: `https://ocw.mit.edu/courses/${id}/`, sourceCode: 'ocw',
  sourceName: 'MIT OpenCourseWare', topics: ['Physics'], levels: ['Undergraduate'], instructors: [], features: [], free: true,
  requiresLogin: false, unified: true, ...extra,
});

const result = (items, extra = {}) => ({
  items, count: items.length, next: null,
  facets: {
    offered_by: [{ key: 'ocw', count: 10 }, { key: 'mitx', count: 3 }],
    level: [{ key: 'undergraduate', count: 8 }],
    topic: [{ key: 'Physics', count: 6 }],
  },
  ...extra,
});

const flush = () => act(async () => { jest.advanceTimersByTime(500); });
const last = () => searchCatalog.mock.calls.at(-1)[0];

beforeEach(() => {
  jest.useFakeTimers();
  searchCatalog.mockReset();
  searchCatalog.mockResolvedValue(result([
    course(1, 'Classical Mechanics'),
    course(2, 'Quantum Physics', { sourceCode: 'mitx', sourceName: 'MITx', requiresLogin: true, unified: false }),
  ]));
});
afterEach(() => jest.useRealTimers());

test('開啟後預設搜尋「全部」來源並顯示課程；需登入的課程有標示', async () => {
  const { getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(last()).toMatchObject({ provider: 'all', q: '' });
  expect(getByText('Classical Mechanics')).toBeTruthy();
  expect(getByText('共 2 門課程')).toBeTruthy();
  expect(getByText(/MITx · 免費 · 需登入/)).toBeTruthy();
  expect(queryByText(/MIT OpenCourseWare · 免費 · 需登入/)).toBeNull();
});

test('來源切換列：全部、MIT、耶魯、陽明交大', async () => {
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  for (const label of ['全部', 'MIT', '耶魯', '陽明交大', '臺大']) expect(getByText(label)).toBeTruthy();
  fireEvent.press(getByText('耶魯'));
  await flush();
  expect(last().provider).toBe('yale');
  fireEvent.press(getByText('陽明交大'));
  await flush();
  expect(last().provider).toBe('nycu');
});

test('MIT 專屬的篩選只在選了 MIT 時出現，切換來源會清除篩選', async () => {
  const { getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(queryByText('有講課影片')).toBeNull();
  fireEvent.press(getByText('MIT'));
  await flush();
  fireEvent.press(getByText('大學部'));
  await flush();
  expect(last()).toMatchObject({ provider: 'mit', level: 'undergraduate' });
  fireEvent.press(getByText('有講課影片'));
  await flush();
  expect(last().hasVideos).toBe(true);
  fireEvent.press(getByText('耶魯'));
  await flush();
  expect(queryByText('有講課影片')).toBeNull();
  expect(last()).toMatchObject({ provider: 'yale', level: null, hasVideos: false });
});

test('輸入關鍵字會（延遲後）搜尋；陽明交大沒有關鍵字時提示輸入', async () => {
  const { getByPlaceholderText, getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  searchCatalog.mockResolvedValue(result([course(9, '微積分甲', { sourceCode: 'nycu', sourceName: '陽明交大 OCW' })], { browse: true, defaultQuery: '微積分' }));
  fireEvent.press(getByText('陽明交大'));
  await flush();
  expect(getByText(/目前顯示的是「微積分」相關課程/)).toBeTruthy();
  searchCatalog.mockClear();
  fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), '微積分');
  await flush();
  expect(last()).toMatchObject({ provider: 'nycu', q: '微積分' });
  expect(queryByText(/目前顯示的是「微積分」相關課程/)).toBeNull();
});

test('捲到底時以來源給的位置載入下一頁並接在後面', async () => {
  searchCatalog.mockResolvedValueOnce(result([course(1, 'A')], { next: 20 }));
  const { getByText, UNSAFE_getByType } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  searchCatalog.mockResolvedValueOnce(result([course(2, 'B')]));
  await act(async () => { UNSAFE_getByType(FlatList).props.onEndReached(); });
  expect(last().cursor).toBe(20);
  expect(getByText('A')).toBeTruthy();
  expect(getByText('B')).toBeTruthy();
});

test('已沒有下一頁時不會再載入', async () => {
  const { UNSAFE_getByType } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  searchCatalog.mockClear();
  await act(async () => { UNSAFE_getByType(FlatList).props.onEndReached(); });
  expect(searchCatalog).not.toHaveBeenCalled();
});

test('部分來源連線失敗時提示，仍顯示其他結果', async () => {
  searchCatalog.mockResolvedValue(result([course(1, 'Classical Mechanics')], { failed: 1 }));
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText(/有 1 個來源暫時無法連線/)).toBeTruthy();
  expect(getByText('Classical Mechanics')).toBeTruthy();
});

test('不開放搜尋的網站會說明改走「網站」分頁', async () => {
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText(/Coursera、edX 等網站不開放自動搜尋/)).toBeTruthy();
});

test('點課程卡片會開啟課程', async () => {
  const onOpen = jest.fn();
  const { getByText } = render(<Explore onOpenCourse={onOpen} />);
  await flush();
  fireEvent.press(getByText('Classical Mechanics'));
  expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }));
});

test('API 失敗時顯示錯誤訊息', async () => {
  searchCatalog.mockRejectedValue(new Error('net'));
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText(/載入失敗/)).toBeTruthy();
});

test('沒有結果時顯示提示', async () => {
  searchCatalog.mockResolvedValue(result([]));
  const { getByText, getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), 'xyz');
  await flush();
  expect(getByText('找不到符合的課程')).toBeTruthy();
});

test('臺大：以背景瀏覽器搜尋（附說明）', async () => {
  const { getByText, getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  fireEvent.press(getByText('臺大'));
  await flush();
  expect(last().provider).toBe('ntu');
  expect(last().browser).toBeDefined(); // 背景瀏覽器（BrowserRunner 的 ref）
  expect(getByText(/臺大網站需要驗證/)).toBeTruthy();
  fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), '微積分');
  await flush();
  expect(last()).toMatchObject({ provider: 'ntu', q: '微積分' });
});

test('臺大驗證未完成或逾時時，顯示專屬的錯誤訊息；被新搜尋取代的不算錯誤', async () => {
  const { getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  searchCatalog.mockRejectedValueOnce(new Error('superseded'));
  fireEvent.press(getByText('臺大'));
  await flush();
  expect(queryByText(/臺大網站沒有回應/)).toBeNull();
  searchCatalog.mockRejectedValue(new Error('timeout'));
  fireEvent.press(getByText('陽明交大'));
  fireEvent.press(getByText('臺大'));
  await flush();
  expect(getByText(/臺大網站沒有回應或驗證未完成/)).toBeTruthy();
});

describe('臺大與陽明交大：沒有關鍵字也有內容可看', () => {
  test('第一次搜尋不帶頁碼（避免臺大被查第 0 頁）', async () => {
    render(<Explore onOpenCourse={() => {}} />);
    await flush();
    expect(last().cursor).toBeUndefined();
  });

  test('沒有關鍵字時提供建議關鍵字，點了就搜尋', async () => {
    const { getByText, getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    fireEvent.press(getByText('陽明交大'));
    await flush();
    expect(getByText('試試看：')).toBeTruthy();
    fireEvent.press(getByText('微積分'));
    await flush();
    expect(last()).toMatchObject({ provider: 'nycu', q: '微積分' });
    expect(getByPlaceholderText('搜尋課程、主題、講師…').props.value).toBe('微積分');
  });

  test('臺大沒有關鍵字時顯示最新與熱門課程，並說明可輸入關鍵字搜尋全部', async () => {
    searchCatalog.mockResolvedValue(result([course('ntu:101S120', '紅樓夢', { sourceCode: 'ntu', sourceName: '臺大開放式課程' })], { browse: true }));
    const { getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    fireEvent.press(getByText('臺大'));
    await flush();
    expect(getByText('紅樓夢')).toBeTruthy();
    expect(getByText(/目前顯示的是最新與熱門課程/)).toBeTruthy();
    expect(queryByText('請輸入關鍵字搜尋臺大課程。')).toBeNull();
  });

  test('有關鍵字但沒有結果時，可回報問題（附查詢網址與解析數量）', async () => {
    const share = jest.spyOn(Share, 'share').mockResolvedValue({});
    searchCatalog.mockResolvedValue(result([], { meta: { url: 'https://ocw.nycu.edu.tw/?s=x', bytes: 1234, parsed: 0 } }));
    const { getByText, getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    fireEvent.press(getByText('陽明交大'));
    fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), 'x');
    await flush();
    fireEvent.press(getByText('回報這個問題'));
    const msg = JSON.parse(share.mock.calls[0][0].message);
    expect(msg).toMatchObject({ provider: 'nycu', query: 'x', meta: { url: 'https://ocw.nycu.edu.tw/?s=x', parsed: 0 } });
  });

  test('有結果時不顯示回報按鈕', async () => {
    const { queryByText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    expect(queryByText('回報這個問題')).toBeNull();
  });
});

describe('沒有搜尋時不顯示「找不到」', () => {
  test('還沒輸入關鍵字、也沒有可顯示的課程時，說明目前沒有內容，而不是「找不到符合的課程」', async () => {
    searchCatalog.mockResolvedValue(result([]));
    const { getByText, queryByText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    expect(queryByText('找不到符合的課程')).toBeNull();
    expect(getByText('目前沒有可顯示的課程')).toBeTruthy();
  });

  test('有關鍵字卻沒有結果時才說找不到', async () => {
    searchCatalog.mockResolvedValue(result([]));
    const { getByText, getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
    await flush();
    fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), '不存在的課');
    await flush();
    expect(getByText('找不到符合的課程')).toBeTruthy();
  });
});
