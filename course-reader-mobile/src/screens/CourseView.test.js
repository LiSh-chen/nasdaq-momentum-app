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
  expect(getByText('介紹')).toBeTruthy();
  expect(getByText('影片 2')).toBeTruthy();
  expect(getByText('講義 2')).toBeTruthy();
  expect(getByText('作業 1')).toBeTruthy();
  expect(getByText('其他頁面 1')).toBeTruthy();
  expect(getByText('內容來源：ocw.example.edu')).toBeTruthy();
});

test('有影片時預設先顯示影片，切到介紹可看課程說明', () => {
  const { getByText, getAllByTestId, queryByText } = setup();
  expect(getAllByTestId('webview')[0].props.source.html).toContain('embed/dQw4w9WgXcQ'); // 預設就是影片播放器
  fireEvent.press(getByText('介紹'));
  expect(getByText('本課程介紹極限、導數與積分的基本概念與應用。')).toBeTruthy();
  expect(queryByText('內容結構')).toBeNull(); // 只列出網頁標題的「內容結構」沒有用，不顯示
});

test('沒有影片時預設顯示第一個有內容的分頁', () => {
  const noVideo = buildContent({
    url: 'https://a.edu/c/', title: 't', heading: '課', headings: [], paras: ['本課程介紹基本概念與應用方式。'],
    links: [{ href: 'https://a.edu/f/ch1.pdf', text: '第一章講義', section: '講義' }],
  });
  const { getByText } = render(<CourseView content={noVideo} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByText('本課程介紹基本概念與應用方式。')).toBeTruthy();
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
  fireEvent.press(getByText('其他頁面 1'));
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
    expect(onOpenChapter).toHaveBeenCalledWith(2, chapters); // 一併傳出章節清單，供上一章／下一章使用
    expect(getByText('影片 2')).toBeTruthy(); // 同一頁的影片、講義分頁仍在
    expect(getByText('講義 2')).toBeTruthy();
  });

  test('上一章／下一章，頭尾章節會停用', () => {
    const onOpenChapter = jest.fn();
    const mid = withChapters(1, onOpenChapter);
    expect(mid.getByText('2 / 3')).toBeTruthy();
    fireEvent.press(mid.getByText('下一章 →'));
    fireEvent.press(mid.getByText('← 上一章'));
    expect(onOpenChapter.mock.calls).toEqual([[2, chapters], [0, chapters]]);
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
    expect(queryByText('其他頁面 1')).toBeNull();
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

test('課程首頁（尚未進入任何章節）有章節時預設顯示章節清單', () => {
  const chapters = [
    { title: '單元一', url: 'https://ocw.example.edu/c/1', section: '單元' },
    { title: '單元二', url: 'https://ocw.example.edu/c/2', section: '單元' },
  ];
  const home = buildContent({ url: 'https://ocw.example.edu/c', title: 't', heading: '課', paras: ['本課程介紹基本概念與應用方式。'] });
  const { getByText } = render(<CourseView content={home} chapters={chapters} onOpenChapter={() => {}} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
  expect(getByText('單元一')).toBeTruthy();
  expect(getByText('單元二')).toBeTruthy();
});

describe('說明文字分區收納', () => {
  const sectioned = buildContent({
    url: 'https://ocw.aca.ntu.edu.tw/courses/111S205', title: 'Calculus (4) - 臺大開放式課程',
    paras: ['This course covers line and surface integrals.', '需修過微積分（一）至（三）。', '指定閱讀：Stewart, Calculus.'],
    paraSections: ['課程概述', '課程要求', '指定閱讀'],
    links: [{ href: 'https://ocw.aca.ntu.edu.tw/files/syllabus.pdf', text: '課程大綱', section: '' }],
  });
  const setupSec = () => render(<CourseView content={sectioned} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);

  test('每個標題一個區塊，只有第一個預設展開', () => {
    const { getByText, queryByText } = setupSec();
    expect(getByText('課程概述')).toBeTruthy();
    expect(getByText('This course covers line and surface integrals.')).toBeTruthy();
    expect(queryByText('需修過微積分（一）至（三）。')).toBeNull();
    expect(queryByText('指定閱讀：Stewart, Calculus.')).toBeNull();
  });

  test('點區塊標題可展開與收合', () => {
    const { getByText, queryByText } = setupSec();
    fireEvent.press(getByText('課程要求'));
    expect(getByText('需修過微積分（一）至（三）。')).toBeTruthy();
    fireEvent.press(getByText('課程概述'));
    expect(queryByText('This course covers line and surface integrals.')).toBeNull();
    fireEvent.press(getByText('課程要求'));
    expect(queryByText('需修過微積分（一）至（三）。')).toBeNull();
  });

  test('只有一個區塊時不需要收合，直接顯示', () => {
    const one = buildContent({ url: 'https://a.edu/c', title: 't', paras: ['本課程介紹基本概念與應用方式。'], paraSections: [''] });
    const { getByText, queryByText } = render(<CourseView content={one} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    expect(getByText('本課程介紹基本概念與應用方式。')).toBeTruthy();
    expect(queryByText('展開')).toBeNull();
  });
});

describe('版面依頁面類型呈現，只留有用的資訊', () => {
  const listing = buildContent({
    url: 'https://ocw.aca.ntu.edu.tw/', title: '臺大開放式課程', heading: '',
    headings: [{ level: 2, text: '本期主打' }, { level: 2, text: 'HIGHLIGHT' }],
    paras: ['臺大線上課程，探索無限可能。'],
    links: [
      { href: 'https://ocw.aca.ntu.edu.tw/about/terms', text: '隱私權政策', section: '', nav: false },
      { href: 'https://ocw.aca.ntu.edu.tw/courses', text: '開始探索', section: '', nav: false },
      { href: 'https://ocw.aca.ntu.edu.tw/courses/114S201', text: '2026/10/01 普通化學丙 彭旭明、鄭原忠 OCW', section: 'HIGHLIGHT', nav: false },
      { href: 'https://ocw.aca.ntu.edu.tw/courses/100S111', text: '微積分 140,993', section: 'HIGHLIGHT', nav: false },
      { href: 'https://ocw.aca.ntu.edu.tw/courses/101S120', text: '紅樓夢 136,890', section: 'HIGHLIGHT', nav: false },
    ],
  });
  const renderListing = (props = {}) => render(<CourseView content={listing} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} {...props} />);

  test('列表型頁面（只有一串可選的項目）：直接顯示清單與提示，沒有分頁列', () => {
    const { getByText, queryByText } = renderListing();
    expect(getByText('請選擇要查看的項目（共 3 項）')).toBeTruthy();
    expect(getByText('普通化學丙 彭旭明、鄭原忠')).toBeTruthy();
    expect(getByText('微積分')).toBeTruthy();
    expect(getByText('紅樓夢')).toBeTruthy();
    // 不需要的資訊
    for (const noise of ['隱私權政策', '開始探索', 'HIGHLIGHT', '本期主打', '內容結構', '臺大線上課程，探索無限可能。', '其他頁面 3', '介紹']) {
      expect(queryByText(noise)).toBeNull();
    }
  });

  test('清單的每一列都有明確可點的樣子（›），點了就整理該頁', () => {
    const onOpenPage = jest.fn();
    const { getByText, getAllByText } = renderListing({ onOpenPage });
    expect(getAllByText('›')).toHaveLength(3);
    fireEvent.press(getByText('微積分'));
    expect(onOpenPage).toHaveBeenCalledWith('https://ocw.aca.ntu.edu.tw/courses/100S111');
  });

  test('只有一個分頁時不顯示分頁列', () => {
    const one = buildContent({ url: 'https://a.edu/c', title: 't', paras: ['本課程介紹基本概念與應用方式。'] });
    const { queryByText, getByText } = render(<CourseView content={one} onClose={() => {}} onOpenPage={() => {}} onOpenSource={() => {}} />);
    expect(getByText('本課程介紹基本概念與應用方式。')).toBeTruthy();
    expect(queryByText('介紹')).toBeNull();
  });

  test('標題列顯示整理過的頁面標題', () => {
    const { getAllByText } = renderListing();
    expect(getAllByText('臺大開放式課程').length).toBeGreaterThan(0);
  });
});
