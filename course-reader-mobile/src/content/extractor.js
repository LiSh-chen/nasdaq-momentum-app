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
    // 選單、側欄、頁尾等版面元素，不算課程內容。
    // 只檢查「內容根節點以內」的祖先：外層包裝容器即使 class 帶有 nav／menu 字樣，也不會誤傷整頁內容。
    var CHROME_CLASS = /(^|[-_])(nav|navbar|navigation|menu|sidebar|breadcrumbs?)([-_]|$)/;
    var isChromeEl = function (n) {
      var tag = n.tagName;
      if (/^(NAV|ASIDE|FOOTER)$/.test(tag)) return true;
      if (tag === 'HEADER' && root === document.body) return true; // 找不到內容區塊時，頁首才算版面
      var role = (n.getAttribute && n.getAttribute('role')) || '';
      if (/navigation|banner|complementary|contentinfo/.test(role)) return true;
      var tokens = (((n.getAttribute && n.getAttribute('class')) || '') + ' ' + (n.id || '')).toLowerCase().split(/\\s+/);
      return tokens.some(function (t) { return CHROME_CLASS.test(t); });
    };
    var inChrome = function (e) {
      for (var n = e; n && n !== root && n !== document.body; n = n.parentElement) {
        if (isChromeEl(n)) return true;
      }
      return false;
    };

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

    // 連結所在那一列的文字（如表格列的「第一週 極限」），用來辨識「線上觀看」這種籠統的連結
    function rowText(a) {
      var row = a.closest ? a.closest('tr,li,dd,p') : null;
      if (!row || row === root) return '';
      var t = text(row).replace(text(a), '').replace(/\\s+/g, ' ').trim();
      return t.length > 80 ? t.slice(0, 80) : t;
    }

    var headings = [].slice.call(root.querySelectorAll('h1,h2,h3,h4')).filter(function (e) { return !inChrome(e); }).map(function (e) {
      return { level: Number(e.tagName.charAt(1)), text: text(e) };
    }).filter(function (h) { return h.text; }).slice(0, 80);

    // 說明文字只取段落與資訊表格；清單項目多半是選單
    var paras = [].slice.call(root.querySelectorAll('p,dd,td,blockquote')).filter(function (e) { return !inChrome(e); }).map(text).filter(function (s) {
      return s.length >= 8 && s.length < 600;
    }).slice(0, 60);

    // 延遲載入的 iframe：src 只是佔位（data:／about:blank），真正的網址在 data-src 等屬性
    var frameSrc = function (e) {
      var src = e.getAttribute('src') || '';
      if (!src || /^(data:|about:)/i.test(src)) {
        src = e.getAttribute('data-src') || e.getAttribute('data-lazy-src') || e.getAttribute('data-lazy') || e.getAttribute('data-original') || src;
      }
      return abs(src);
    };
    var frames = [].slice.call(document.querySelectorAll('iframe')).map(function (e) {
      return { src: frameSrc(e), title: e.title || '', section: sectionOf(e) };
    }).filter(function (f) { return /^https?:/.test(f.src); });

    var videos = [].slice.call(document.querySelectorAll('video')).map(function (v) {
      var s = v.currentSrc || v.src || ((v.querySelector('source') || {}).src) || '';
      return { src: abs(s), title: v.title || '', section: sectionOf(v) };
    }).filter(function (v) { return v.src; });

    var links = [].slice.call(document.querySelectorAll('a[href]')).map(function (a) {
      return { href: abs(a.getAttribute('href')), text: text(a), section: sectionOf(a), context: rowText(a), nav: inChrome(a) };
    }).filter(function (l) { return /^https?:/.test(l.href); }).slice(0, 800);

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
