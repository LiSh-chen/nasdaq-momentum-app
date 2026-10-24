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
