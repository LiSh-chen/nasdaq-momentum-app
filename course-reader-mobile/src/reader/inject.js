// 注入 OCW 課程頁：只保留課程主體內容，套用本 App 統一的排版與深色模式
const READER_CSS = `
:root { color-scheme: light dark; }
body { margin: 0 !important; padding: 12px 14px 48px !important; background: #fff !important; color: #111827 !important;
  font: 16px/1.7 -apple-system, system-ui, 'Noto Sans TC', sans-serif !important; }
header, nav, footer, .navbar, #course-nav, .course-nav { display: none !important; }
img, video { max-width: 100% !important; height: auto !important; }
iframe[src*="youtube"], iframe[src*="archive.org"] { width: 100% !important; aspect-ratio: 16 / 9; height: auto !important; border: 0; }
table { display: block; overflow-x: auto; }
pre, code { overflow-x: auto; white-space: pre-wrap; }
h1, h2, h3 { line-height: 1.3; }
a { color: #2563eb; }
@media (prefers-color-scheme: dark) {
  body { background: #0b1220 !important; color: #e5e7eb !important; }
  a { color: #93c5fd; }
}
`;

export const READER_JS = `
(function () {
  try {
    if (!document.querySelector('meta[name=viewport]')) {
      var vp = document.createElement('meta');
      vp.name = 'viewport';
      vp.content = 'width=device-width, initial-scale=1';
      document.head.appendChild(vp);
    }
    var st = document.createElement('style');
    st.textContent = ${JSON.stringify(READER_CSS)};
    document.head.appendChild(st);
    var main = document.querySelector('#course-main-content');
    var n = main;
    while (n && n !== document.body) {
      var p = n.parentElement;
      if (!p) break;
      Array.prototype.forEach.call(p.children, function (c) {
        if (c !== n && !/^(SCRIPT|STYLE|LINK)$/.test(c.tagName)) c.style.display = 'none';
      });
      n = p;
    }
  } catch (e) {}
  true;
})();
`;
