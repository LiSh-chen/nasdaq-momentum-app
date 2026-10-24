import { decodeEntities, stripHtml, tidyTitle } from './text';

test('解碼常見的 HTML 實體（含截圖中出現的 &nbsp;）', () => {
  expect(decodeEntities("class's&nbsp;Solidworks, Creo&nbsp;and NX")).toBe("class's Solidworks, Creo and NX");
  expect(decodeEntities('A &amp; B &lt;C&gt; &quot;D&quot; &#39;E&#39; &#x4e2d;&ndash;&hellip;')).toBe('A & B <C> "D" \'E\' 中–…');
  expect(decodeEntities('&unknown;')).toBe('&unknown;');
});

test('stripHtml 去標籤、解碼實體並整理空白', () => {
  expect(stripHtml('<p>Hello&nbsp;<b>world</b></p>\n<p>again</p>')).toBe('Hello world again');
  expect(stripHtml(null)).toBe('');
});

test('tidyTitle：去掉日期前綴、OCW 後綴與瀏覽次數，保留課名', () => {
  expect(tidyTitle('2026/10/01 普通化學丙 彭旭明、鄭原忠 OCW')).toBe('普通化學丙 彭旭明、鄭原忠');
  expect(tidyTitle('2025/11/22 AI 時代人類的獨特性 葉素玲 MOOCOCW')).toBe('AI 時代人類的獨特性 葉素玲');
  expect(tidyTitle('微積分 2,034,407')).toBe('微積分');
  expect(tidyTitle('Introduction to Phonetics I (With Pronunciation Practicum) 120,205')).toBe('Introduction to Phonetics I (With Pronunciation Practicum)');
  expect(tidyTitle('Calculus 2024')).toBe('Calculus 2024'); // 年份不是瀏覽次數
  expect(tidyTitle('OCW')).toBe('OCW'); // 清完會變空時保留原文
  expect(tidyTitle('  微積分  ')).toBe('微積分');
});
