const ENTITIES = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—',
  hellip: '…', rsquo: '’', lsquo: '‘', ldquo: '“', rdquo: '”', copy: '©',
};

export const decodeEntities = (s) =>
  s
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&([a-z]+);/gi, (m, name) => ENTITIES[name.toLowerCase()] ?? m);

export const stripHtml = (s) => decodeEntities(String(s ?? '').replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();

// 課程連結的標題常帶有裝飾：日期前綴（2026/10/01）、站名後綴（OCW／MOOCOCW）、瀏覽次數（140,993）
export function tidyTitle(title) {
  const t = String(title ?? '').replace(/\s+/g, ' ').trim();
  const out = t
    .replace(/^\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\s+/, '')
    .replace(/\s*(MOOC)?OCW$/, '')
    .replace(/\s+\d{1,3}(,\d{3})+$/, '')
    .trim();
  return out || t;
}
