// 已整理過的頁面保留在記憶體，再次開啟（例如回到上一章）時不必重新載入
const MAX = 60;
const store = new Map();

export const getCached = (url) => store.get(url);
export const dropCached = (url) => store.delete(url);
export function setCached(url, content) {
  store.delete(url);
  store.set(url, content);
  if (store.size > MAX) store.delete(store.keys().next().value);
}
export const clearCache = () => store.clear();
