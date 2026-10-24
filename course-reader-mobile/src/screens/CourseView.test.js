import React from 'react';
import { Linking } from 'react-native';
import { render, fireEvent } from '@testing-library/react-native';
import CourseView from './CourseView';
import { buildContent } from '../content/build';

const content = buildContent({
  url: 'https://ocw.example.edu/course/101/', title: 't', heading: '微積分（一）',
  headings: [{ level: 2, text: '課程大綱' }],
  paras: ['本課程介紹極限、導數與積分的基本概念與應用。'],
  frames: [
    { src: 'https://www.youtube.com/embed/dQw4w9WgXcQ', title: '第一講', section: '影片' },
    { src: 'https://www.youtube.com/embed/9bZkp7q19f0', title: '第二講', section: '影片' },
  ],
  links: [
    { href: 'https://ocw.example.edu/files/ch1.pdf', text: '第一章講義', section: '講義' },
    { href: 'https://ocw.example.edu/files/data.zip', text: '資料集', section: '講義' },
    { href: 'https://ocw.example.edu/files/hw1.pdf', text: '作業一', section: '作業' },
    { href: 'https://ocw.example.edu/course/101/week-2/', text: '第二週', section: '' },
  ],
});

const setup = (props = {}) => render(<CourseView content={content} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} {...props} />);

beforeEach(() => jest.spyOn(Linking, 'openURL').mockResolvedValue());

test('顯示課程標題、各分類分頁與數量、來源', () => {
  const { getByText } = setup();
  expect(getByText('微積分（一）')).toBeTruthy();
  expect(getByText('大綱')).toBeTruthy();
  expect(getByText('影片 2')).toBeTruthy();
  expect(getByText('講義 2')).toBeTruthy();
  expect(getByText('作業 1')).toBeTruthy();
  expect(getByText('相關頁面 1')).toBeTruthy();
  expect(getByText('內容來源：ocw.example.edu')).toBeTruthy();
});

test('有影片時預設先顯示影片，切到大綱可看課程說明', () => {
  const { getByText, getAllByTestId } = setup();
  expect(getAllByTestId('webview')[0].props.source.html).toContain('embed/dQw4w9WgXcQ'); // 預設就是影片播放器
  fireEvent.press(getByText('大綱'));
  expect(getByText('課程說明')).toBeTruthy();
  expect(getByText('課程大綱')).toBeTruthy();
});

test('沒有影片時預設顯示第一個有內容的分頁', () => {
  const noVideo = buildContent({
    url: 'https://a.edu/c/', title: 't', heading: '課', headings: [], paras: ['本課程介紹基本概念與應用方式。'],
    links: [{ href: 'https://a.edu/f/ch1.pdf', text: '第一章講義', section: '講義' }],
  });
  const { getByText } = render(<CourseView content={noVideo} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByText('課程說明')).toBeTruthy();
});

test('影片分頁：以來源網域嵌入播放，可切換影片', () => {
  const { getByText, getAllByTestId } = setup();
  fireEvent.press(getByText('影片 2'));
  let players = getAllByTestId('webview');
  expect(players[0].props.source.baseUrl).toBe('https://ocw.example.edu');
  expect(players[0].props.source.html).toContain('embed/dQw4w9WgXcQ');
  fireEvent.press(getByText('第二講'));
  players = getAllByTestId('webview');
  expect(players[0].props.source.html).toContain('embed/9bZkp7q19f0');
});

test('影片無法播放時可改在 YouTube 開啟', () => {
  const { getByText } = setup();
  fireEvent.press(getByText('影片 2'));
  fireEvent.press(getByText('影片無法播放？在 YouTube 開啟'));
  expect(Linking.openURL).toHaveBeenCalledWith('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
});

test('講義分頁：PDF 在 App 內檢視，其他檔案交給系統下載', () => {
  const { getByText, getAllByTestId } = setup();
  fireEvent.press(getByText('講義 2'));
  fireEvent.press(getByText('資料集'));
  expect(Linking.openURL).toHaveBeenCalledWith('https://ocw.example.edu/files/data.zip');
  fireEvent.press(getByText('第一章講義'));
  const uris = getAllByTestId('webview').map((w) => w.props.source.uri).filter(Boolean);
  expect(uris[0]).toContain('docs.google.com/gview');
  expect(uris[0]).toContain(encodeURIComponent('https://ocw.example.edu/files/ch1.pdf'));
});

test('相關頁面可繼續整理，原頁面按鈕交給外層', () => {
  const onOpenPage = jest.fn();
  const onOpenSource = jest.fn();
  const { getByText } = setup({ onOpenPage, onOpenSource });
  fireEvent.press(getByText('相關頁面 1'));
  fireEvent.press(getByText('第二週'));
  expect(onOpenPage).toHaveBeenCalledWith('https://ocw.example.edu/course/101/week-2/');
  fireEvent.press(getByText('原頁面'));
  expect(onOpenSource).toHaveBeenCalledWith('https://ocw.example.edu/course/101/');
});

test('頁面沒有可整理的內容時顯示說明', () => {
  const empty = buildContent({ url: 'https://a.b/', title: '空白頁' });
  const { getByText } = render(<CourseView content={empty} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByText('這個頁面沒有找到影片、講義或大綱。')).toBeTruthy();
});

describe('多章節課程', () => {
  const chapters = [
    { title: 'Syllabus', url: 'https://ocw.example.edu/c/1/', section: 'Syllabus' },
    { title: 'Lesson 1', url: 'https://ocw.example.edu/c/2/', section: 'Week 1' },
    { title: 'Lesson 2', url: 'https://ocw.example.edu/c/3/', section: 'Week 1' },
  ];
  const withChapters = (index, onOpenChapter = () => {}) =>
    render(<CourseView content={content} chapters={chapters} chapterIndex={index} onOpenChapter={onOpenChapter} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);

  test('章節分頁排第一，列出所有章節並標示目前這章', () => {
    const { getByText } = withChapters(1);
    expect(getByText('章節 3')).toBeTruthy();
    fireEvent.press(getByText('章節 3'));
    expect(getByText('▶ Lesson 1')).toBeTruthy();
    expect(getByText('Lesson 2')).toBeTruthy();
  });

  test('點章節會開啟該章，且每章都用同一套分類呈現', () => {
    const onOpenChapter = jest.fn();
    const { getByText } = withChapters(0, onOpenChapter);
    fireEvent.press(getByText('章節 3'));
    fireEvent.press(getByText('Lesson 2'));
    expect(onOpenChapter).toHaveBeenCalledWith(2);
    expect(getByText('影片 2')).toBeTruthy(); // 同一頁的影片、講義分頁仍在
    expect(getByText('講義 2')).toBeTruthy();
  });

  test('上一章／下一章，頭尾章節會停用', () => {
    const onOpenChapter = jest.fn();
    const mid = withChapters(1, onOpenChapter);
    expect(mid.getByText('2 / 3')).toBeTruthy();
    fireEvent.press(mid.getByText('下一章 →'));
    fireEvent.press(mid.getByText('← 上一章'));
    expect(onOpenChapter.mock.calls).toEqual([[2], [0]]);
    mid.unmount();

    const first = withChapters(0, onOpenChapter);
    onOpenChapter.mockClear();
    fireEvent.press(first.getByText('← 上一章'));
    expect(onOpenChapter).not.toHaveBeenCalled();
    first.unmount();

    const last = withChapters(2, onOpenChapter);
    fireEvent.press(last.getByText('下一章 →'));
    expect(onOpenChapter).not.toHaveBeenCalled();
  });

  test('標題顯示該章標題；有章節時不顯示相關頁面分頁', () => {
    const { getByText, queryByText } = withChapters(1);
    expect(getByText('Lesson 1')).toBeTruthy();
    expect(queryByText('相關頁面 1')).toBeNull();
  });

  test('這一章沒有內容時顯示章節，不會是空白畫面', () => {
    const empty = buildContent({ url: 'https://ocw.example.edu/c/1/', title: 'Syllabus' });
    const { getByText } = render(<CourseView content={empty} chapters={chapters} chapterIndex={0} onOpenChapter={() => {}} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    expect(getByText('▶ Syllabus')).toBeTruthy();
  });
});

test('沒有內容時提供重新整理（例如登入後回來重試）', () => {
  const empty = buildContent({ url: 'https://a.b/', title: '空白頁' });
  const onRefresh = jest.fn();
  const { getByText } = render(<CourseView content={empty} onRefresh={onRefresh} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  fireEvent.press(getByText('重新整理'));
  expect(onRefresh).toHaveBeenCalled();
});

test('資源頁型檔案：點開時先解析實際檔案網址，再於 App 內檢視', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue({ ok: true, text: async () => '<a href="/static/ps1.pdf">Download</a>' });
  const c = buildContent({
    url: 'https://ocw.example.edu/courses/c1/', title: 't', heading: '課',
    links: [{ href: 'https://ocw.example.edu/courses/c1/resources/ps1_pdf/', text: 'Problem Set 1', section: 'Assignments' }],
  });
  const { getByText, findAllByTestId } = render(<CourseView content={c} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  fireEvent.press(getByText('Problem Set 1'));
  const views = await findAllByTestId('webview');
  expect(views.map((v) => v.props.source.uri).filter(Boolean)[0]).toContain(encodeURIComponent('https://ocw.example.edu/static/ps1.pdf'));
});
