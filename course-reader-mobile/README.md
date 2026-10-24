# 課程閱讀器（手機版）

Expo / React Native。搜尋、篩選各家免費課程，並以**統一的閱讀介面**在 App 內閱讀，不跳轉外部瀏覽器；需登入的來源則用 App 內網頁開啟並記住登入。

## 畫面

- **探索**：關鍵字搜尋，篩選來源／程度／主題，可只看「有講課影片」或依熱門排序，無限捲動載入。
- **課程詳情**：簡介、講師、章節清單；收藏；「開始／繼續閱讀」。
- **閱讀器**（開放授權的 OCW 課程）：只顯示課程內容，統一排版與深色模式；上一頁／下一頁、章節目錄、進度列；課程內影片直接在頁內播放；外部連結才交給系統瀏覽器。
- **我的課程**：繼續閱讀（自動記住讀到哪一頁）與收藏。
- **網站**：自訂網站清單，每站記住登入（沿用先前版本）。
- **整理**：在「網站」分頁（或閱讀器）按工具列的「整理」，會在 App 內載入目前頁面，擷取其中的影片、講義、作業、大綱，歸類後用統一版面呈現（大綱／影片／講義／作業／考試／閱讀／其他／相關頁面）。YouTube 影片直接在 App 內播放，PDF 與 Office 檔在 App 內檢視。適用於臺大 OCW 等沒有公開目錄 API 的網站。

## 資料來源（「探索」可搜尋）

| 來源 | 取得方式 | 備註 |
| --- | --- | --- |
| MIT（OCW、MITx、xPRO…） | MIT Learn 公開 API | 可依來源、程度、主題、有講課影片篩選 |
| 耶魯開放課程 | 讀取課程列表頁，本機篩選 | 各講次自動成為章節 |
| 陽明交大 OCW | 網站搜尋頁（`?s=關鍵字`） | 支援翻頁 |
| 臺大開放式課程 | **App 內建瀏覽器**在背景載入搜尋頁 | 網站有人機驗證，必須由真實瀏覽器載入；只在選「臺大」時執行；需要驗證時才會顯示網頁 |

「全部」會同時查 MIT、耶魯、陽明交大（不含臺大，因為較慢）。

**不提供自動搜尋的網站**：Coursera（`robots.txt` 禁止）、edX／Khan／均一／ShareCourse（結果由前端產生，沒有可查詢的公開介面）。這些仍可在「網站」分頁瀏覽後按「整理」。

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

## 章節

課程頁底下的子頁面會自動成為「章節」（有上一章／下一章）：OCW 與耶魯有專屬的目錄解析；其他網站（如臺大 `/courses/代碼/單元編號`）則取頁面正下一層的連結。

## 頁面整理器的運作

1. `ExtractScreen` 以 WebView 載入頁面（與一般瀏覽相同，因此能通過網站的人機驗證與登入）。
2. 載入完成後注入 `src/content/extractor.js`，擷取標題、`<iframe>`／`<video>`、所有連結（並標註各自所在區塊的標題）與說明文字，用 `postMessage` 回傳。
3. `src/content/build.js` 辨識 YouTube（含播放清單）、mp4、檔案；`classify.js` 依標題、檔名、所在區塊歸類。
4. `CourseView` 以固定的分頁呈現。影片以來源網站的網域嵌入，與原頁面條件一致；無法播放時提供「在 YouTube 開啟」。
5. 擷取是通用規則，不針對單一網站寫死。若某網站整理結果不理想，用檢視畫面下方的「分享診斷資訊」取得頁面結構，再針對該站調整。

## 已知限制

- 閱讀器靠頁面中的 `#course-main-content` 擷取主體並隱藏其餘版面；OCW 改版時需更新 `src/reader/inject.js`。
- 較新的 OCW 課程首頁沒有章節連結，此時直接開啟課程首頁。
- 不儲存帳號密碼，只保留網站發給瀏覽器的登入狀態。
