import { decodeEntities, stripHtml } from './text';

test('解碼常見的 HTML 實體（含截圖中出現的 &nbsp;）', () => {
  expect(decodeEntities("class's&nbsp;Solidworks, Creo&nbsp;and NX")).toBe("class's Solidworks, Creo and NX");
  expect(decodeEntities('A &amp; B &lt;C&gt; &quot;D&quot; &#39;E&#39; &#x4e2d;&ndash;&hellip;')).toBe('A & B <C> "D" \'E\' 中–…');
  expect(decodeEntities('&unknown;')).toBe('&unknown;');
});

test('stripHtml 去標籤、解碼實體並整理空白', () => {
  expect(stripHtml('<p>Hello&nbsp;<b>world</b></p>\n<p>again</p>')).toBe('Hello world again');
  expect(stripHtml(null)).toBe('');
});
