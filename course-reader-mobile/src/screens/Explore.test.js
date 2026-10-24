import React from 'react';
import { render, fireEvent, act } from '@testing-library/react-native';
import Explore from './Explore';
import { searchCourses } from '../api/mitLearn';

jest.mock('../api/mitLearn', () => ({
  ...jest.requireActual('../api/mitLearn'),
  searchCourses: jest.fn(),
}));

const course = (id, title, extra = {}) => ({
  id, title, description: '', image: null, url: `https://ocw.mit.edu/courses/${id}/`, sourceCode: 'ocw',
  sourceName: 'MIT OpenCourseWare', topics: ['Physics'], levels: ['Undergraduate'], instructors: [], features: [], free: true, unified: true, ...extra,
});

const result = (items, extra = {}) => ({
  items, count: items.length, hasMore: false,
  facets: {
    offered_by: [{ key: 'ocw', count: 10 }, { key: 'mitx', count: 3 }],
    level: [{ key: 'undergraduate', count: 8 }],
    topic: [{ key: 'Physics', count: 6 }],
  },
  ...extra,
});

const flush = () => act(async () => { jest.advanceTimersByTime(500); });

beforeEach(() => {
  jest.useFakeTimers();
  searchCourses.mockReset();
  searchCourses.mockResolvedValue(result([course(1, 'Classical Mechanics'), course(2, 'Quantum Physics', { sourceCode: 'mitx', unified: false })]));
});
afterEach(() => jest.useRealTimers());

test('開啟後載入課程並顯示篩選選項', async () => {
  const { getByText, getAllByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText('Classical Mechanics')).toBeTruthy();
  expect(getByText('共 2 門課程')).toBeTruthy();
  expect(getAllByText('MIT OpenCourseWare', { exact: false }).length).toBeGreaterThanOrEqual(2); // 篩選標籤 + 課程卡片
  expect(getByText('大學部')).toBeTruthy();
  expect(getByText('MITx')).toBeTruthy();
});

test('輸入關鍵字會（延遲後）以 q 搜尋', async () => {
  const { getByPlaceholderText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  searchCourses.mockClear();
  fireEvent.changeText(getByPlaceholderText('搜尋課程、主題、講師…'), 'quantum');
  await flush();
  expect(searchCourses).toHaveBeenLastCalledWith(expect.objectContaining({ q: 'quantum', offset: 0 }));
});

test('點選篩選會重新搜尋，再點一次取消', async () => {
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  fireEvent.press(getByText('大學部'));
  await flush();
  expect(searchCourses).toHaveBeenLastCalledWith(expect.objectContaining({ level: 'undergraduate' }));
  fireEvent.press(getByText('大學部'));
  await flush();
  expect(searchCourses).toHaveBeenLastCalledWith(expect.objectContaining({ level: null }));
  fireEvent.press(getByText('有講課影片'));
  await flush();
  expect(searchCourses).toHaveBeenLastCalledWith(expect.objectContaining({ hasVideos: true }));
  fireEvent.press(getByText('MITx'));
  await flush();
  expect(searchCourses).toHaveBeenLastCalledWith(expect.objectContaining({ offeredBy: 'mitx' }));
});

test('點課程卡片會開啟課程', async () => {
  const onOpen = jest.fn();
  const { getByText } = render(<Explore onOpenCourse={onOpen} />);
  await flush();
  fireEvent.press(getByText('Classical Mechanics'));
  expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }));
});

test('API 失敗時顯示錯誤訊息', async () => {
  searchCourses.mockRejectedValue(new Error('net'));
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText(/載入失敗/)).toBeTruthy();
});

test('沒有結果時顯示提示', async () => {
  searchCourses.mockResolvedValue(result([]));
  const { getByText } = render(<Explore onOpenCourse={() => {}} />);
  await flush();
  expect(getByText('找不到符合的課程')).toBeTruthy();
});
