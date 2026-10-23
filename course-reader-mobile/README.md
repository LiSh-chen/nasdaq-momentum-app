# 課程閱讀器（手機版）

Expo / React Native。以 `react-native-webview` 在 App 內直接閱讀各校免費課程，不跳轉外部瀏覽器，並保留各站登入。

## 執行

`@react-native-cookies/cookies` 含原生模組，**無法在 Expo Go 執行**，需建置 dev client：

```bash
cd course-reader-mobile
npm install
npx expo run:android   # 或 npx expo run:ios（需 macOS + Xcode）
```

## 運作方式

- 啟用 `sharedCookiesEnabled` / `thirdPartyCookiesEnabled` / `domStorageEnabled`，登入 cookie 由系統 WebView 保存。
- 每次頁面載入完成，會把沒有到期日的 session cookie 延長為 30 天並 flush 到磁碟，重開 App 仍維持登入。
- 「登出」只清除目前網站網域的 cookie；長按清單中的網站可移除並清除其登入資料。
- Android 返回鍵：先在網頁內返回，沒有上一頁時回到網站清單。
- 網站清單存於 AsyncStorage，預設清單與桌面版共用 `sites.default.json`。

## 注意

- 不儲存帳號密碼，只保留網站發給瀏覽器的登入狀態。
- 手機上所有網站共用同一個 cookie 儲存區（以網域區分），不像桌面版是完全獨立分區。
- 請遵守各平台服務條款，僅供個人學習。

## 測試

```bash
npm test   # Jest 元件測試（WebView 與 CookieManager 以 mock 取代）
```
