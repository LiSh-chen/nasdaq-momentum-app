import { resolveFileUrl } from './resolve';

const ok = (html) => jest.fn().mockResolvedValue({ ok: true, text: async () => html });

test('從資源頁找出實際的檔案網址，並轉為絕對網址', async () => {
  const f = ok('<a href="/nav">x</a><a href="/courses/c1/abc123_ps1.pdf?dl=1&amp;v=2">Download file</a>');
  expect(await resolveFileUrl('https://ocw.example.edu/courses/c1/resources/ps1_pdf/', f)).toBe('https://ocw.example.edu/courses/c1/abc123_ps1.pdf?dl=1&v=2');
});

test('找不到檔案連結時回傳 null', async () => {
  expect(await resolveFileUrl('https://a.b/r/', ok('<a href="/x">x</a>'))).toBeNull();
});

test('頁面載入失敗時丟出錯誤', async () => {
  await expect(resolveFileUrl('https://a.b/r/', jest.fn().mockResolvedValue({ ok: false, status: 404 }))).rejects.toThrow('404');
});
