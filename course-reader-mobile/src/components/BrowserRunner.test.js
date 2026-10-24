import React, { createRef } from 'react';
import { StyleSheet } from 'react-native';
import { render, act, fireEvent } from '@testing-library/react-native';
import BrowserRunner from './BrowserRunner';

const flat = (el) => StyleSheet.flatten(el.props.style);
const send = (getByTestId, msg) => act(async () => { getByTestId('webview').props.onMessage({ nativeEvent: { data: JSON.stringify(msg) } }); });
const content = { type: 'content', url: 'https://x.edu/', links: [], paras: [] };

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

function setup() {
  const ref = createRef();
  const utils = render(<BrowserRunner ref={ref} />);
  return { ref, ...utils };
}

test('沒有工作時不渲染任何東西', () => {
  const { queryByTestId } = setup();
  expect(queryByTestId('runner')).toBeNull();
  expect(queryByTestId('webview')).toBeNull();
});

test('run：以隱藏的瀏覽器載入網址（共用 cookie），擷取完成後回傳結果並收起瀏覽器', async () => {
  const { ref, getByTestId, queryByTestId } = setup();
  let result;
  act(() => { ref.current.run('https://x.edu/search?q=1').then((r) => { result = r; }); });
  expect(getByTestId('webview').props.source.uri).toBe('https://x.edu/search?q=1');
  expect(getByTestId('webview').props.sharedCookiesEnabled).toBe(true);
  expect(flat(getByTestId('runner')).opacity).toBe(0);
  expect(getByTestId('runner').props.pointerEvents).toBe('none');
  await send(getByTestId, content);
  expect(result).toEqual(content);
  expect(queryByTestId('runner')).toBeNull();
});

test('網站要求驗證時才把網頁顯示出來，通過後自動繼續並收起', async () => {
  const { ref, getByTestId, getByText, queryByTestId } = setup();
  let result;
  act(() => { ref.current.run('https://x.edu/').then((r) => { result = r; }); });
  await send(getByTestId, { type: 'challenge' });
  expect(flat(getByTestId('runner')).opacity).toBeUndefined();
  expect(getByText(/網站正在進行驗證/)).toBeTruthy();
  await act(async () => { jest.advanceTimersByTime(1600); });
  await send(getByTestId, content);
  expect(result).toEqual(content);
  expect(queryByTestId('runner')).toBeNull();
});

test('使用者可取消驗證，Promise 以 cancelled 結束', async () => {
  const { ref, getByTestId, getByText } = setup();
  let error;
  act(() => { ref.current.run('https://x.edu/').catch((e) => { error = e; }); });
  await send(getByTestId, { type: 'challenge' });
  await act(async () => { fireEvent.press(getByText('取消')); });
  expect(error.message).toBe('cancelled');
});

test('ready：內容還沒出現就多等幾次再擷取，出現後才完成', async () => {
  const { ref, getByTestId } = setup();
  let result;
  const ready = jest.fn((m) => m.links.length > 0);
  act(() => { ref.current.run('https://x.edu/', { ready }).then((r) => { result = r; }); });
  await send(getByTestId, content); // 還沒有結果
  expect(result).toBeUndefined();
  await act(async () => { jest.advanceTimersByTime(1600); });
  const filled = { ...content, links: [{ href: 'https://x.edu/a' }] };
  await send(getByTestId, filled);
  expect(result).toEqual(filled);
});

test('ready 一直不成立時，多次重試後仍交出目前的內容（例如真的沒有結果）', async () => {
  const { ref, getByTestId } = setup();
  let result;
  act(() => { ref.current.run('https://x.edu/', { ready: () => false }).then((r) => { result = r; }); });
  for (let i = 0; i < 7; i++) await send(getByTestId, content);
  expect(result).toEqual(content);
});

test('逾時會丟出 timeout；新的工作會取代舊的（superseded）', async () => {
  const { ref } = setup();
  let timeout;
  act(() => { ref.current.run('https://x.edu/a', { timeoutMs: 1000 }).catch((e) => { timeout = e; }); });
  await act(async () => { jest.advanceTimersByTime(1100); });
  expect(timeout.message).toBe('timeout');

  let first;
  act(() => { ref.current.run('https://x.edu/1').catch((e) => { first = e; }); });
  act(() => { ref.current.run('https://x.edu/2'); });
  await act(async () => {}); // 讓 Promise 的 catch 處理器有機會執行
  expect(first.message).toBe('superseded');
});

test('頁面載入完成後會注入擷取腳本；錯誤訊息會讓 Promise 失敗', async () => {
  const { ref, getByTestId } = setup();
  let error;
  act(() => { ref.current.run('https://x.edu/').catch((e) => { error = e; }); });
  await send(getByTestId, { type: 'error', message: 'boom' });
  expect(error.message).toBe('boom');
});
