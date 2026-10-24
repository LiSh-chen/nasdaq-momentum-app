import React from 'react';
import { render, act } from '@testing-library/react-native';
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
