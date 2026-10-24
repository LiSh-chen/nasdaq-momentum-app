import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CourseDetail from './CourseDetail';
import { fetchOutline } from '../api/ocw';

jest.mock('../api/ocw', () => ({ ...jest.requireActual('../api/ocw'), fetchOutline: jest.fn() }));

const course = (extra = {}) => ({
  id: 7, title: 'Classical Mechanics', description: '力學入門', image: null, url: 'https://ocw.mit.edu/courses/8-01sc/',
  sourceCode: 'ocw', sourceName: 'MIT OpenCourseWare', topics: ['Physics'], levels: ['Undergraduate'],
  instructors: ['Prof. A'], features: ['Lecture Videos'], free: true, unified: true, ...extra,
});

const outline = [
  { key: 'syllabus', title: 'Syllabus', url: 'https://ocw.mit.edu/courses/8-01sc/pages/syllabus/', items: [] },
  { key: 'week-1', title: 'Week 1', url: 'https://ocw.mit.edu/courses/8-01sc/pages/week-1/', items: [{ title: 'Lesson 1', url: 'https://ocw.mit.edu/courses/8-01sc/pages/week-1/l1/' }] },
];

beforeEach(async () => {
  await AsyncStorage.clear();
  fetchOutline.mockReset();
});

test('載入章節後可開始閱讀，頁面依章節順序攤平', async () => {
  fetchOutline.mockResolvedValue(outline);
  const onRead = jest.fn();
  const { getByText, findByText } = render(<CourseDetail course={course()} onRead={onRead} />);
  fireEvent.press(await findByText('開始學習'));
  const [c, pages, start] = onRead.mock.calls[0];
  expect(c.id).toBe(7);
  expect(pages.map((p) => p.title)).toEqual(['Syllabus', 'Week 1', 'Lesson 1']);
  expect(start).toBe(0);
  expect(getByText('1 個單元')).toBeTruthy();
});

test('點章節標題會從該章節開始閱讀', async () => {
  fetchOutline.mockResolvedValue(outline);
  const onRead = jest.fn();
  const { findByText } = render(<CourseDetail course={course()} onRead={onRead} />);
  fireEvent.press(await findByText('Week 1'));
  expect(onRead.mock.calls[0][2]).toBe(1);
});

test('沒有章節目錄時退回課程首頁並顯示說明', async () => {
  fetchOutline.mockResolvedValue([]);
  const onRead = jest.fn();
  const { findByText } = render(<CourseDetail course={course()} onRead={onRead} />);
  expect(await findByText('這門課沒有章節目錄，將直接開啟課程首頁。')).toBeTruthy();
  fireEvent.press(await findByText('開始學習'));
  expect(onRead.mock.calls[0][1]).toEqual([{ title: 'Classical Mechanics', url: 'https://ocw.mit.edu/courses/8-01sc/', section: 'Classical Mechanics' }]);
});

test('章節載入失敗時仍可從課程首頁閱讀', async () => {
  fetchOutline.mockRejectedValue(new Error('net'));
  const { findByText } = render(<CourseDetail course={course()} onRead={() => {}} />);
  expect(await findByText('章節載入失敗，仍可從課程首頁開始閱讀。')).toBeTruthy();
});

test('其他來源（如需登入的 MITx）也走同一個流程：以課程首頁開始，不抓章節目錄', async () => {
  const onRead = jest.fn();
  const c = course({ sourceCode: 'mitx', sourceName: 'MITx', unified: false, url: 'https://courses.example/x' });
  const { getByText } = render(<CourseDetail course={c} onRead={onRead} />);
  fireEvent.press(getByText('開始學習'));
  expect(onRead).toHaveBeenCalledWith(expect.objectContaining({ id: 7 }), [{ title: 'Classical Mechanics', url: 'https://courses.example/x', section: 'Classical Mechanics' }], 0);
  expect(fetchOutline).not.toHaveBeenCalled();
});

test('有閱讀紀錄時顯示繼續學習並從上次的章節開始', async () => {
  fetchOutline.mockResolvedValue(outline);
  await AsyncStorage.setItem('progress', JSON.stringify({ 7: { index: 2, title: 'Lesson 1', total: 3, updatedAt: 1, course: {} } }));
  const onRead = jest.fn();
  const { findByText } = render(<CourseDetail course={course()} onRead={onRead} />);
  fireEvent.press(await findByText('繼續學習（第 3 章）'));
  expect(onRead.mock.calls[0][2]).toBe(2);
});

test('收藏會寫入儲存並切換狀態', async () => {
  fetchOutline.mockResolvedValue(outline);
  const { findByText, getByText } = render(<CourseDetail course={course()} onRead={() => {}} />);
  fireEvent.press(await findByText('☆ 收藏'));
  await waitFor(() => expect(getByText('★ 已收藏')).toBeTruthy());
  expect(Object.keys(JSON.parse(await AsyncStorage.getItem('favorites')))).toEqual(['7']);
});
