import React from 'react';
import { Alert } from 'react-native';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CookieManager from '@react-native-cookies/cookies';
import App from './App';
import defaultSites from './sites.default.json';

beforeEach(async () => {
  await AsyncStorage.clear();
  jest.clearAllMocks();
  CookieManager.get.mockResolvedValue({});
});

test('首次開啟顯示預設網站清單', () => {
  const { getByText } = render(<App />);
  expect(getByText(defaultSites[0].name)).toBeTruthy();
  expect(getByText(defaultSites[5].name)).toBeTruthy();
});

test('點選網站後在 WebView 內載入，不跳轉', () => {
  const { getByText, getByTestId } = render(<App />);
  fireEvent.press(getByText('MIT OpenCourseWare'));
  expect(getByTestId('webview').props.source.uri).toBe('https://ocw.mit.edu/');
  expect(getByTestId('webview').props.sharedCookiesEnabled).toBe(true);
  expect(getByTestId('webview').props.setSupportMultipleWindows).toBe(false);
});

test('新增自訂網站並寫入儲存', async () => {
  const { getByText, getByPlaceholderText, queryByPlaceholderText } = render(<App />);
  fireEvent.press(getByText('＋ 新增網站'));
  fireEvent.changeText(getByPlaceholderText('網站名稱'), '測試大學');
  fireEvent.changeText(getByPlaceholderText('https://...'), 'https://example.edu/');
  fireEvent.press(getByText('新增'));
  // 新增的網站排在清單最後（FlatList 測試環境只渲染前 10 筆），改驗證已寫入儲存且對話框已關閉
  await waitFor(() => expect(queryByPlaceholderText('網站名稱')).toBeNull());
  const saved = JSON.parse(await AsyncStorage.getItem('sites'));
  expect(saved).toHaveLength(defaultSites.length + 1);
  expect(saved.at(-1)).toMatchObject({ name: '測試大學', url: 'https://example.edu/' });
});

test('網址格式錯誤會被拒絕', () => {
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => {});
  const { getByText, getByPlaceholderText } = render(<App />);
  fireEvent.press(getByText('＋ 新增網站'));
  fireEvent.changeText(getByPlaceholderText('網站名稱'), 'X');
  fireEvent.changeText(getByPlaceholderText('https://...'), 'not a url');
  fireEvent.press(getByText('新增'));
  expect(alert).toHaveBeenCalledWith('網址格式不正確');
});

test('頁面載入完成後，把 session cookie 延長為持久 cookie', async () => {
  CookieManager.get.mockResolvedValue({
    sid: { name: 'sid', value: 'abc' },
    keep: { name: 'keep', value: 'x', expires: '2030-01-01T00:00:00.000Z' },
  });
  const { getByText, getByTestId } = render(<App />);
  fireEvent.press(getByText('edX'));
  await act(async () => {
    getByTestId('webview').props.onNavigationStateChange({
      loading: false, url: 'https://www.edx.org/', canGoBack: false, canGoForward: false,
    });
  });
  expect(CookieManager.set).toHaveBeenCalledTimes(1);
  expect(CookieManager.set.mock.calls[0][1]).toMatchObject({ name: 'sid', expires: expect.any(String) });
  expect(CookieManager.flush).toHaveBeenCalled();
});

test('登出只清除當前網站的 cookie', async () => {
  CookieManager.get.mockResolvedValue({ sid: { name: 'sid', value: 'abc' } });
  jest.spyOn(Alert, 'alert').mockImplementation((_t, _m, buttons) => buttons.find((b) => b.style === 'destructive').onPress());
  const { getByText } = render(<App />);
  fireEvent.press(getByText('Coursera'));
  await act(async () => { fireEvent.press(getByText('登出')); });
  expect(CookieManager.get).toHaveBeenCalledWith('https://www.coursera.org/');
  expect(CookieManager.set.mock.calls[0][1].expires).toBe(new Date(0).toISOString());
});
