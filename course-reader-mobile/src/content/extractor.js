// 注入到任何課程頁面，擷取標題、影片、檔案連結、大綱文字，並標註每個項目所在區塊的標題。
// 結果以 postMessage 回傳；分類與整理在 App 端（build.js）完成。
export const EXTRACT_JS = `
(function () {
  function post(o) { window.ReactNativeWebView.postMessage(JSON.stringify(o)); }
  try {
    var t = document.title || '';
    if (/just a moment|請稍候|attention required|verify you are human|checking your browser/i.test(t)) {
      post({ type: 'challenge' });
      return;
    }
    var abs = function (u) { try { return new URL(u, location.href).href; } catch (e) { return ''; } };
    var text = function (e) { return (e.textContent || '').replace(/\\s+/g, ' ').trim(); };
    var root = document.querySelector('#course-main-content, main, article, [role=main], #content, .content') || document.body;
    var isHeading = function (e) { return e && /^H[1-6]$/.test(e.tagName); };

    // 往上、往前找最近的標題，作為這個項目的「所在區塊」
    function sectionOf(el) {
      var n = el;
      while (n && n !== document.body) {
        var p = n.previousElementSibling;
        while (p) {
          if (isHeading(p)) return text(p);
          var hs = p.querySelectorAll ? p.querySelectorAll('h1,h2,h3,h4,h5,h6') : [];
          if (hs.length) return text(hs[hs.length - 1]);
          p = p.previousElementSibling;
        }
        n = n.parentElement;
      }
      return '';
    }

    var headings = [].slice.call(root.querySelectorAll('h1,h2,h3,h4')).map(function (e) {
      return { level: Number(e.tagName.charAt(1)), text: text(e) };
    }).filter(function (h) { return h.text; }).slice(0, 80);

    var paras = [].slice.call(root.querySelectorAll('p,li,td')).map(text).filter(function (s) {
      return s.length >= 8 && s.length < 600;
    }).slice(0, 60);

    var frames = [].slice.call(document.querySelectorAll('iframe')).map(function (e) {
      return { src: abs(e.getAttribute('src') || e.getAttribute('data-src') || ''), title: e.title || '', section: sectionOf(e) };
    }).filter(function (f) { return f.src; });

    var videos = [].slice.call(document.querySelectorAll('video')).map(function (v) {
      var s = v.currentSrc || v.src || ((v.querySelector('source') || {}).src) || '';
      return { src: abs(s), title: v.title || '', section: sectionOf(v) };
    }).filter(function (v) { return v.src; });

    var links = [].slice.call(document.querySelectorAll('a[href]')).map(function (a) {
      return { href: abs(a.getAttribute('href')), text: text(a), section: sectionOf(a) };
    }).filter(function (l) { return /^https?:/.test(l.href); }).slice(0, 500);

    var h1 = document.querySelector('h1');
    post({
      type: 'content', url: location.href, title: t, heading: h1 ? text(h1) : '',
      lang: document.documentElement.lang || '', headings: headings, paras: paras,
      frames: frames, videos: videos, links: links
    });
  } catch (e) {
    post({ type: 'error', message: String(e && e.message || e) });
  }
})();
true;
`;
