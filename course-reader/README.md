# 課程閱讀器

在單一 App 內閱讀各大專院校與機構的免費線上課程，不會跳轉到外部瀏覽器，並記住各站登入。

## 執行

```bash
cd course-reader
npm install
npm start
```

## 運作方式

- 每個網站使用獨立的 `persist:site-<id>` 分區，cookie 與 localStorage 互不干擾，關閉 App 後仍保留。
- 沒有到期日的 session cookie 會被延長為 30 天，避免每次重開都要重新登入。
- 站內 `target=_blank` / `window.open` 一律在同一分頁開啟。
- 內嵌的是完整瀏覽器視圖（非 iframe），因此不受 `X-Frame-Options` 限制。
- 可新增自訂網站；「登出此站」只清除該站資料。

## 注意

- 本 App 不儲存你的帳號密碼，只保存網站發給瀏覽器的登入狀態；各站若有登入期限，到期仍需重新登入。
- 請遵守各課程平台的服務條款，僅用於個人學習。
