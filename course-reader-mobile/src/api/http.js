// 讀取公開網頁。使用明確的 User-Agent 與逾時；不處理、不繞過任何網站的驗證機制。
const UA = 'CourseReader/0.1 (personal study app)';

export async function getText(url, fetchImpl = fetch, timeoutMs = 15000) {
  const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  try {
    const res = await fetchImpl(url, { headers: { 'User-Agent': UA, Accept: 'text/html,*/*' }, signal: ctrl?.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
  } finally {
    if (timer) clearTimeout(timer);
  }
}
