import React from 'react';
import { Linking } from 'react-native';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Reader from './Reader';
import { READER_JS } from '../reader/inject';

const course = {
  id: 7, title: 'Classical Mechanics', url: 'https://ocw.mit.edu/courses/8-01sc/', sourceCode: 'ocw', sourceName: 'MIT OpenCourseWare',
  topics: [], levels: [], instructors: [], features: [], free: true, unified: true, description: '', image: null,
};
const pages = [
  { title: 'Syllabus', url: 'https://ocw.mit.edu/courses/8-01sc/pages/syllabus/', section: 'Syllabus' },
  { title: 'Week 1', url: 'https://ocw.mit.edu/courses/8-01sc/pages/week-1/', section: 'Week 1' },
  { title: 'Lesson 1', url: 'https://ocw.mit.edu/courses/8-01sc/pages/week-1/l1/', section: 'Week 1' },
];

beforeEach(async () => {
  await AsyncStorage.clear();
  jest.spyOn(Linking, 'openURL').mockResolvedValue();
});

test('載入起始頁，並注入統一排版的腳本', () => {
  const { getByTestId, getAllByText } = render(<Reader course={course} pages={pages} startIndex={1} onClose={() => {}} />);
  expect(getByTestId('webview').props.source.uri).toBe(pages[1].url);
  expect(getByTestId('webview').props.injectedJavaScript).toBe(READER_JS);
  expect(getAllByText('Week 1').length).toBeGreaterThan(0);
  expect(READER_JS).toContain('#course-main-content');
});

test('上一頁／下一頁切換頁面並儲存閱讀進度', async () => {
  const { getByText, getByTestId } = render(<Reader course={course} pages={pages} onClose={() => {}} />);
  expect(getByText('1 / 3')).toBeTruthy();
  fireEvent.press(getByText('下一頁 →'));
  expect(getByTestId('webview').props.source.uri).toBe(pages[1].url);
  expect(getByText('2 / 3')).toBeTruthy();
  await waitFor(async () => {
    const p = JSON.parse(await AsyncStorage.getItem('progress'));
    expect(p[7]).toMatchObject({ index: 1, title: 'Week 1', total: 3 });
  });
});

test('第一頁不能再往前、最後一頁不能再往後', () => {
  const first = render(<Reader course={course} pages={pages} startIndex={0} onClose={() => {}} />);
  fireEvent.press(first.getByText('← 上一頁'));
  expect(first.getByText('1 / 3')).toBeTruthy();
  first.unmount();
  const last = render(<Reader course={course} pages={pages} startIndex={2} onClose={() => {}} />);
  fireEvent.press(last.getByText('下一頁 →'));
  expect(last.getByText('3 / 3')).toBeTruthy();
});

test('點到目錄中的頁面在閱讀器內切換；課程內其他頁面直接載入；外部連結用系統瀏覽器', () => {
  const { getByTestId, getByText } = render(<Reader course={course} pages={pages} onClose={() => {}} />);
  const should = getByTestId('webview').props.onShouldStartLoadWithRequest;

  // 目前頁面與影片播放器的子框架
  expect(should({ url: pages[0].url })).toBe(true);
  expect(should({ url: 'https://www.youtube.com/embed/abc', isTopFrame: false })).toBe(true);

  // 目錄中的其他頁面：不走網頁導覽，改由閱讀器切頁
  expect(should({ url: pages[2].url + '#top' })).toBe(false);
  expect(getByText('3 / 3')).toBeTruthy();

  // 同一門課、不在目錄中的頁面（如影片資源頁）：直接在 WebView 載入
  const inner = getByTestId('webview').props.onShouldStartLoadWithRequest;
  expect(inner({ url: 'https://ocw.mit.edu/courses/8-01sc/resources/lecture-1/' })).toBe(true);

  // 外部連結與其他課程
  expect(inner({ url: 'https://example.com/' })).toBe(false);
  expect(inner({ url: 'https://ocw.mit.edu/courses/other-course/' })).toBe(false);
  expect(Linking.openURL).toHaveBeenCalledWith('https://example.com/');
  expect(Linking.openURL).toHaveBeenCalledWith('https://ocw.mit.edu/courses/other-course/');
});

test('目錄可開啟並跳到指定頁面', () => {
  const { getByText, getAllByText } = render(<Reader course={course} pages={pages} onClose={() => {}} />);
  fireEvent.press(getByText('目錄'));
  fireEvent.press(getAllByText('Lesson 1').at(-1));
  expect(getByText('3 / 3')).toBeTruthy();
});
