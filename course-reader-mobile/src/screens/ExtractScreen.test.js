import React from 'react';
import { render, act, fireEvent } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import ExtractScreen from './ExtractScreen';

const send = (getByTestId, msg) => act(() => { getByTestId('webview').props.onMessage({ nativeEvent: { data: JSON.stringify(msg) } }); });

const raw = {
  type: 'content', url: 'https://ocw.example.edu/course/101/', title: 't', heading: '微積分（一）', lang: 'zh-TW',
  headings: [], paras: [], videos: [],
  frames: [{ src: 'https://www.youtube.com/embed/dQw4w9WgXcQ', title: '第一講', section: '影片' }],
  links: [{ href: 'https://ocw.example.edu/files/ch1.pdf', text: '第一章講義', section: '講義' }],
};

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('載入頁面（使用共用 cookie），整理完成後顯示統一檢視', async () => {
  const { getByTestId, getByText, queryByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByTestId('webview').props.source.uri).toBe(raw.url);
  expect(getByTestId('webview').props.sharedCookiesEnabled).toBe(true);
  expect(getByText('整理頁面中…')).toBeTruthy();
  await send(getByTestId, raw);
  expect(queryByText('整理頁面中…')).toBeNull();
  expect(getByText('微積分（一）')).toBeTruthy();
  expect(getByText('影片 1')).toBeTruthy();
  expect(getByText('講義 1')).toBeTruthy();
});

test('遇到網站驗證時提示並自動重試', async () => {
  const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  await send(getByTestId, { type: 'challenge' });
  expect(getByText(/網站正在進行驗證/)).toBeTruthy();
  await act(async () => { jest.advanceTimersByTime(1600); });
  await send(getByTestId, raw);
  expect(getByText('微積分（一）')).toBeTruthy();
});

test('整理失敗時顯示原因與重試', async () => {
  const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  await send(getByTestId, { type: 'error', message: 'boom' });
  expect(getByText('無法整理此頁：boom')).toBeTruthy();
  expect(getByText('重試')).toBeTruthy();
});

test('忽略格式不正確的訊息', async () => {
  const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  await act(async () => { getByTestId('webview').props.onMessage({ nativeEvent: { data: 'not json' } }); });
  expect(getByText('整理頁面中…')).toBeTruthy();
});

describe('課程流程', () => {
  const course = { id: 9, title: 'Calculus', url: raw.url, sourceCode: 'ocw', sourceName: 'MIT OCW', topics: [], levels: [], instructors: [], features: [], free: true, unified: true };
  const chapters = [{ title: '第一章', url: raw.url, section: 'A' }, { title: '第二章', url: raw.url + 'b/', section: 'A' }];

  test('整理完成後記錄閱讀進度（章節索引與總章節數）', async () => {
    await AsyncStorage.clear();
    const { getByTestId, getByText } = render(
      <ExtractScreen url={raw.url} course={course} chapters={chapters} chapterIndex={1} onClose={() => {}} onOpenChapter={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />
    );
    await send(getByTestId, raw);
    expect(getByText('章節 2')).toBeTruthy();
    const saved = JSON.parse(await AsyncStorage.getItem('progress'));
    expect(saved[9]).toMatchObject({ index: 1, title: '第二章', total: 2 });
  });

  test('單頁應用程式一開始沒內容時多等幾次，內容出現就顯示', async () => {
    const empty = { ...raw, frames: [], links: [] };
    const { getByTestId, getByText, queryByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    await send(getByTestId, empty);
    await act(async () => { jest.advanceTimersByTime(2100); });
    expect(queryByText('這個頁面沒有找到影片、講義或大綱。')).toBeNull(); // 還在等
    await send(getByTestId, raw);
    expect(getByText('影片 1')).toBeTruthy();
  });

  test('始終沒有內容時顯示空狀態，按重新整理會重新載入頁面', async () => {
    const empty = { ...raw, frames: [], links: [] };
    const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    for (let i = 0; i < 3; i++) {
      await send(getByTestId, empty);
      await act(async () => { jest.advanceTimersByTime(2100); });
    }
    await send(getByTestId, empty);
    expect(getByText('這個頁面沒有找到影片、講義或大綱。')).toBeTruthy();
    fireEvent.press(getByText('重新整理'));
    expect(getByText('整理頁面中…')).toBeTruthy();
    expect(getByTestId('webview').props.source.uri).toBe(raw.url);
  });
});
