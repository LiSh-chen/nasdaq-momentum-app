import React from 'react';
import { StyleSheet } from 'react-native';
import { render, act, fireEvent } from '@testing-library/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import ExtractScreen from './ExtractScreen';
import { clearCache } from '../content/cache';

const send = (getByTestId, msg) => act(() => { getByTestId('webview').props.onMessage({ nativeEvent: { data: JSON.stringify(msg) } }); });

const raw = {
  type: 'content', url: 'https://ocw.example.edu/course/101/', title: 't', heading: '微積分（一）', lang: 'zh-TW',
  headings: [], paras: [], videos: [],
  frames: [{ src: 'https://www.youtube.com/embed/dQw4w9WgXcQ', title: '第一講', section: '影片' }],
  links: [{ href: 'https://ocw.example.edu/files/ch1.pdf', text: '第一章講義', section: '講義' }],
};

beforeEach(() => { clearCache(); jest.useFakeTimers(); });
afterEach(() => jest.useRealTimers());

test('載入頁面（使用共用 cookie），整理完成後顯示統一檢視', async () => {
  const { getByTestId, getByText, queryByTestId } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByTestId('webview').props.source.uri).toBe(raw.url);
  expect(getByTestId('webview').props.sharedCookiesEnabled).toBe(true);
  expect(getByTestId('loading')).toBeTruthy();
  await send(getByTestId, raw);
  expect(queryByTestId('loading')).toBeNull();
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
  const { getByTestId } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  await act(async () => { getByTestId('webview').props.onMessage({ nativeEvent: { data: 'not json' } }); });
  expect(getByTestId('loading')).toBeTruthy();
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
    expect(getByTestId('loading')).toBeTruthy();
    expect(getByTestId('webview').props.source.uri).toBe(raw.url);
  });
});

describe('不讓使用者看到轉換過程', () => {
  const flat = (el) => StyleSheet.flatten(el.props.style);

  test('載入期間原網頁隱藏（仍在背景執行），只顯示載入畫面與課程名稱', () => {
    const chapters = [{ title: '第一章', url: raw.url, section: 'A' }, { title: '第二章', url: raw.url + 'b/', section: 'A' }];
    const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} chapters={chapters} chapterIndex={1} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    expect(flat(getByTestId('webview')).opacity).toBe(0);
    expect(getByTestId('webview').props.pointerEvents).toBe('none');
    expect(getByTestId('loading')).toBeTruthy();
    expect(getByText('第二章')).toBeTruthy();
  });

  test('網站要求驗證時才把網頁顯示出來，讓使用者操作', async () => {
    const { getByTestId, queryByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    await send(getByTestId, { type: 'challenge' });
    expect(flat(getByTestId('webview')).opacity).toBeUndefined();
    expect(queryByTestId('loading')).toBeNull();
    expect(getByText(/網站正在進行驗證/)).toBeTruthy();
  });

  test('整理失敗時顯示網頁與原因', async () => {
    const { getByTestId, getByText } = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    await send(getByTestId, { type: 'error', message: 'boom' });
    expect(flat(getByTestId('webview')).opacity).toBeUndefined();
    expect(getByText('無法整理此頁：boom')).toBeTruthy();
  });

  test('已整理過的頁面再次開啟時直接顯示結果，不重新載入網頁', async () => {
    const first = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    await send(first.getByTestId, raw);
    first.unmount();
    const second = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    // 沒有載入原頁面的 WebView（畫面上剩下的 WebView 是影片播放器）
    expect(second.queryAllByTestId('webview').some((w) => w.props.source.uri === raw.url)).toBe(false);
    expect(second.queryByTestId('loading')).toBeNull();
    expect(second.getByText('影片 1')).toBeTruthy();
  });

  test('重新整理會捨棄快取重新載入', async () => {
    const first = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    await send(first.getByTestId, { ...raw, frames: [], links: [] }); // 空結果不快取
    for (let i = 0; i < 3; i++) await act(async () => { jest.advanceTimersByTime(2100); });
    await send(first.getByTestId, { ...raw, frames: [], links: [] });
    first.unmount();
    const second = render(<ExtractScreen url={raw.url} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    expect(second.getByTestId('loading')).toBeTruthy(); // 空結果沒有被快取
  });
});

describe('子頁面自動成為章節', () => {
  const home = {
    type: 'content', url: 'https://ocw.aca.ntu.edu.tw/courses/100S111', title: '微積分 - 臺大開放式課程', heading: '微積分', lang: 'zh-Hant',
    headings: [], paras: ['本課程介紹微積分的基本概念與應用。'], frames: [], videos: [],
    links: [
      { href: 'https://ocw.aca.ntu.edu.tw/courses/100S111/58', text: '第15章 向量場', section: '', nav: false },
      { href: 'https://ocw.aca.ntu.edu.tw/courses/100S111/59', text: '第16章 向量微積分', section: '', nav: false },
    ],
  };

  test('課程首頁底下有多個子頁面時顯示章節，點選時把清單一併交出', async () => {
    const onOpenChapter = jest.fn();
    const { getByTestId, getByText } = render(<ExtractScreen url={home.url} onClose={() => {}} onOpenPage={() => {}} onOpenChapter={onOpenChapter} onOpenSource={() => {}} />);
    await send(getByTestId, home);
    expect(getByText('章節 2')).toBeTruthy();
    fireEvent.press(getByText('第16章 向量微積分'));
    expect(onOpenChapter).toHaveBeenCalledWith(1, [
      { title: '第15章 向量場', url: 'https://ocw.aca.ntu.edu.tw/courses/100S111/58', section: '單元' },
      { title: '第16章 向量微積分', url: 'https://ocw.aca.ntu.edu.tw/courses/100S111/59', section: '單元' },
    ]);
  });

  test('只有一個子頁面時不算章節', async () => {
    const one = { ...home, links: home.links.slice(0, 1) };
    const { getByTestId, queryByText } = render(<ExtractScreen url={home.url} onClose={() => {}} onOpenPage={() => {}} onOpenChapter={() => {}} onOpenSource={() => {}} />);
    await send(getByTestId, one);
    expect(queryByText(/^章節/)).toBeNull();
  });
});
