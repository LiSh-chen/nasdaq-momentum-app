// 資源頁 → 實際檔案網址（OCW 的講義、作業連到的是含下載連結的頁面）
const FILE_HREF = /href="([^"]+\.(?:pdf|pptx?|docx?|xlsx?|zip)(?:\?[^"]*)?)"/gi;

export async function resolveFileUrl(pageUrl, fetchImpl = fetch) {
  const res = await fetchImpl(pageUrl);
  if (!res.ok) throw new Error(`resolve ${res.status}`);
  const html = await res.text();
  for (const m of html.matchAll(FILE_HREF)) {
    try {
      return new URL(m[1].replace(/&amp;/g, '&'), pageUrl).href;
    } catch {}
  }
  return null;
}
