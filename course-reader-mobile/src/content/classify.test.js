import { classifyMaterial } from './classify';

const c = (title, section = '', url = '') => classifyMaterial({ title, section, url });

test('依標題分類（中英文）', () => {
  expect(c('第三週講義')).toBe('lecture');
  expect(c('Lecture Notes 3')).toBe('lecture');
  expect(c('投影片 Week 2')).toBe('lecture');
  expect(c('作業一')).toBe('assignment');
  expect(c('Problem Set 4')).toBe('assignment');
  expect(c('期中考試題')).toBe('exam');
  expect(c('Final Exam')).toBe('exam');
  expect(c('參考書目')).toBe('reading');
  expect(c('課程大綱')).toBe('overview');
  expect(c('Syllabus')).toBe('overview');
  expect(c('其他資料')).toBe('other');
});

test('考試優先於作業（例如 exam solutions）', () => {
  expect(c('Exam 1 Solutions')).toBe('exam');
  expect(c('Midterm problem set')).toBe('exam');
});

test('標題看不出來時，改看檔名，再看所在區塊的標題', () => {
  expect(c('下載', '', 'https://x.edu/files/hw3.pdf')).toBe('assignment');
  expect(c('第 5 章', '講義', 'https://x.edu/files/ch5.pdf')).toBe('lecture');
  expect(c('第 5 章', '', 'https://x.edu/files/ch5.pdf')).toBe('other');
});

test('檔案自己的名稱優先於所在區塊', () => {
  expect(c('Lecture 3 slides', 'Assignments')).toBe('lecture');
});
