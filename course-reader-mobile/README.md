# 課程閱讀器（手機版）

Expo / React Native。搜尋、篩選各家免費課程，並以**統一的閱讀介面**在 App 內閱讀，不跳轉外部瀏覽器；需登入的來源則用 App 內網頁開啟並記住登入。

## 畫面

- **探索**：關鍵字搜尋，篩選來源／程度／主題，可只看「有講課影片」或依熱門排序，無限捲動載入。
- **課程詳情**：簡介、講師、章節清單；收藏；「開始／繼續閱讀」。
- **閱讀器**（開放授權的 OCW 課程）：只顯示課程內容，統一排版與深色模式；上一頁／下一頁、章節目錄、進度列；課程內影片直接在頁內播放；外部連結才交給系統瀏覽器。
- **我的課程**：繼續閱讀（自動記住讀到哪一頁）與收藏。
- **網站**：自訂網站清單，每站記住登入（沿用先前版本）。

## 資料來源

| 來源 | 目錄 | 閱讀 |
| --- | --- | --- |
| MIT OpenCourseWare（CC 授權） | MIT Learn 公開 API（免金鑰） | 統一閱讀介面 |
| MITx、xPRO、MITPE 等 | 同上 | App 內網頁（需登入，登入狀態會保留） |

課程內容受版權與服務條款保護、且需登入的平台（Coursera、edX、Khan Academy 等），不擷取內容重新排版，只能在「網站」分頁以內嵌網頁使用。臺大 OCW 有 Cloudflare 人機驗證，同樣只能用內嵌網頁。

## 執行

`@react-native-cookies/cookies` 含原生模組，**無法在 Expo Go 執行**，需建置 dev client：

```bash
cd course-reader-mobile
npm install
npx expo run:android   # 或 npx expo run:ios（需 macOS + Xcode）
```

CI 會自動建置 APK（`.github/workflows/build-android-apk.yml`），到 Actions 該次執行頁面底部下載 `course-reader-apk`。

## 測試

```bash
npm test                                                  # 元件與解析單元測試（網路以 mock 取代）
node --experimental-default-type=module scripts/live-check.mjs   # 對真實 MIT Learn API / OCW 網站驗證解析（需網路）
```

`live-check` 也由 `.github/workflows/live-check.yml` 在相關檔案變動時執行。

## 已知限制

- 閱讀器靠頁面中的 `#course-main-content` 擷取主體並隱藏其餘版面；OCW 改版時需更新 `src/reader/inject.js`。
- 較新的 OCW 課程首頁沒有章節連結，此時直接開啟課程首頁。
- 不儲存帳號密碼，只保留網站發給瀏覽器的登入狀態。
