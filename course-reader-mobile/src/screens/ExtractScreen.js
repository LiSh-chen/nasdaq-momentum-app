import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { buildContent, tabsOf } from '../content/build';
import { dropCached, getCached, setCached } from '../content/cache';
import { EXTRACT_JS } from '../content/extractor';
import { persistSessionCookies } from '../cookies';
import { saveProgress } from '../store';
import { Btn, colors } from '../ui';
import CourseView from './CourseView';

const MAX_TRIES = 8;

// 在 App 內以一般瀏覽器載入頁面（通過網站的人機驗證與登入），再擷取內容並整理成統一格式。
// 載入期間原網頁是隱藏的，只顯示載入畫面；只有需要使用者操作（驗證）或失敗時才顯示網頁。
export default function ExtractScreen({ url, course, chapters, chapterIndex, onClose, onOpenPage, onOpenChapter, onOpenSource }) {
  const web = useRef(null);
  const tries = useRef(0);
  const emptyTries = useRef(0);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState(() => {
    const cached = getCached(url);
    return cached ? { status: 'done', content: cached, message: '' } : { status: 'loading', content: null, message: '' };
  });

  const chapter = chapters?.[chapterIndex];
  const run = () => web.current?.injectJavaScript(EXTRACT_JS);

  // 整理完成（或取自快取）時記錄閱讀進度
  useEffect(() => {
    if (state.status !== 'done' || !course) return;
    saveProgress(course, { index: chapterIndex ?? 0, title: chapter?.title ?? state.content.title, url, total: chapters?.length ?? 1 });
  }, [state.status]);

  const onMessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.nativeEvent.data); } catch { return; }
    if (msg.type === 'challenge') {
      setState((s) => (s.status === 'done' ? s : { ...s, status: 'challenge' }));
      if (++tries.current < MAX_TRIES) setTimeout(run, 1500);
      else setState({ status: 'error', content: null, message: '網站的驗證尚未通過，請在畫面中完成驗證後重試。' });
    } else if (msg.type === 'content') {
      const content = buildContent(msg);
      // 單頁應用程式（如需登入的課程平台）載入後才渲染內容；沒抓到東西時多等幾次
      if (!tabsOf(content).length && emptyTries.current < 3) {
        emptyTries.current += 1;
        setTimeout(run, 2000);
        return;
      }
      content.diagnostic = {
        url: msg.url, title: msg.title, heading: msg.heading, lang: msg.lang,
        counts: { headings: msg.headings.length, paras: msg.paras.length, frames: msg.frames.length, videos: msg.videos.length, links: msg.links.length },
        headings: msg.headings.slice(0, 40), frames: msg.frames.slice(0, 20), videos: msg.videos.slice(0, 10),
        links: msg.links.slice(0, 120), built: { videos: content.videos.length, materials: content.materials.length, related: content.related.length },
      };
      if (tabsOf(content).length) setCached(url, content); // 空結果不快取，登入後重試才會重新抓
      setState({ status: 'done', content, message: '' });
      persistSessionCookies(msg.url);
    } else if (msg.type === 'error') {
      setState({ status: 'error', content: null, message: msg.message });
    }
  };

  const reset = () => {
    dropCached(url);
    tries.current = 0;
    emptyTries.current = 0;
    setState({ status: 'loading', content: null, message: '' });
    setAttempt((a) => a + 1);
  };

  if (state.status === 'done') {
    return (
      <CourseView
        content={state.content}
        chapters={chapters}
        chapterIndex={chapterIndex}
        onClose={onClose}
        onOpenPage={onOpenPage}
        onOpenChapter={onOpenChapter}
        onOpenSource={onOpenSource}
        onRefresh={reset}
      />
    );
  }

  // 需要使用者操作（網站驗證）或出錯時，才把網頁顯示出來
  const showWeb = state.status === 'challenge' || state.status === 'error';
  const title = chapter?.title ?? course?.title ?? '課程';

  return (
    <SafeAreaView style={styles.root} edges={[]}>
      <View style={styles.bar}>
        <Btn label="✕" onPress={onClose} />
        <Text style={styles.title} numberOfLines={1}>{showWeb ? '請完成網站驗證' : '整理中…'}</Text>
        {state.status === 'error' ? <Btn label="重試" onPress={reset} /> : <View style={{ width: 52 }} />}
      </View>

      {showWeb ? (
        <View style={styles.status}>
          <Text style={styles.statusText}>
            {state.status === 'challenge' ? '網站正在進行驗證，通過後會自動整理；若出現勾選框請完成驗證。' : `無法整理此頁：${state.message}`}
          </Text>
        </View>
      ) : null}

      <View style={{ flex: 1 }}>
        <WebView
          key={attempt}
          ref={web}
          source={{ uri: url }}
          sharedCookiesEnabled
          thirdPartyCookiesEnabled
          domStorageEnabled
          javaScriptEnabled
          setSupportMultipleWindows={false}
          onLoadEnd={() => { tries.current = 0; emptyTries.current = 0; setTimeout(run, 800); }}
          onMessage={onMessage}
          style={showWeb ? styles.web : styles.hiddenWeb}
          pointerEvents={showWeb ? 'auto' : 'none'}
        />
        {!showWeb ? (
          <View style={styles.loading} testID="loading">
            <ActivityIndicator size="large" color={colors.primary} />
            <Text style={styles.loadingTitle} numberOfLines={2}>{title}</Text>
            <Text style={styles.loadingHint}>正在整理影片、講義與大綱…</Text>
          </View>
        ) : null}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.dark },
  bar: { flexDirection: 'row', alignItems: 'center', height: 48, paddingHorizontal: 6 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center' },
  status: { paddingHorizontal: 14, paddingBottom: 8 },
  statusText: { color: '#9ca3af', fontSize: 12 },
  web: { flex: 1 },
  // 仍在載入與執行，只是看不見
  hiddenWeb: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, opacity: 0 },
  loading: { ...StyleSheet.absoluteFillObject, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center', gap: 14, padding: 32 },
  loadingTitle: { fontSize: 16, fontWeight: '600', color: colors.text, textAlign: 'center' },
  loadingHint: { fontSize: 12, color: colors.muted },
});
