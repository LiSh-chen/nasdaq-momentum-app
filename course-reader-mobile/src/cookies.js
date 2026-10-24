import CookieManager from '@react-native-cookies/cookies';

// 沒有到期日的 session cookie 在 App 被關閉後會消失；延長後登入才能跨次啟動保留
const COOKIE_LIFETIME_MS = 30 * 24 * 60 * 60 * 1000;

export async function persistSessionCookies(url) {
  try {
    const cookies = await CookieManager.get(url);
    const expires = new Date(Date.now() + COOKIE_LIFETIME_MS).toISOString();
    await Promise.all(
      Object.values(cookies)
        .filter((c) => !c.expires)
        .map((c) => CookieManager.set(url, { ...c, expires }))
    );
    await CookieManager.flush();
  } catch {}
}

export async function clearCookiesFor(url) {
  try {
    const cookies = await CookieManager.get(url);
    const past = new Date(0).toISOString();
    await Promise.all(
      Object.values(cookies).map((c) => CookieManager.set(url, { ...c, expires: past }))
    );
    await CookieManager.flush();
  } catch {}
}
