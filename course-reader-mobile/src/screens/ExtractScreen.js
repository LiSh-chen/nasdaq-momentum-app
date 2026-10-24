import React, { useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { buildContent, tabsOf } from '../content/build';
import { EXTRACT_JS } from '../content/extractor';
import { persistSessionCookies } from '../cookies';
import { saveProgress } from '../store';
import { Btn, colors } from '../ui';
import CourseView from './CourseView';

const MAX_TRIES = 8;

// 在 App 內以一般瀏覽器載入頁面（通過網站的人機驗證與登入），再擷取內容並整理成統一格式
export default function ExtractScreen({ url, course, chapters, chapterIndex, onClose, onOpenPage, onOpenChapter, onOpenSource }) {
  const web = useRef(null);
  const tries = useRef(0);
  const emptyTries = useRef(0);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState({ status: 'loading', content: null, message: '' });

  const run = () => web.current?.injectJavaScript(EXTRACT_JS);

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
      setState({ status: 'done', content, message: '' });
      persistSessionCookies(msg.url);
      if (course) {
        const ch = chapters?.[chapterIndex];
        saveProgress(course, { index: chapterIndex ?? 0, title: ch?.title ?? content.title, url: msg.url, total: chapters?.length ?? 1 });
      }
    } else if (msg.type === 'error') {
      setState({ status: 'error', content: null, message: msg.message });
    }
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
        onRefresh={() => { tries.current = 0; emptyTries.current = 0; setState({ status: 'loading', content: null, message: '' }); setAttempt((a) => a + 1); }}
      />
    );
  }

  return (
    <SafeAreaView style={styles.root} edges={[]}>
      <View style={styles.bar}>
        <Btn label="✕" onPress={onClose} />
        <Text style={styles.title} numberOfLines={1}>整理頁面中…</Text>
        {state.status === 'error' ? <Btn label="重試" onPress={() => { tries.current = 0; setState({ status: 'loading', content: null, message: '' }); setAttempt((a) => a + 1); }} /> : <View style={{ width: 52 }} />}
      </View>
      <View style={styles.status}>
        {state.status !== 'error' ? <ActivityIndicator color="#fff" /> : null}
        <Text style={styles.statusText}>
          {state.status === 'challenge' ? '網站正在進行驗證，通過後會自動整理；若出現勾選框請完成驗證。'
            : state.status === 'error' ? `無法整理此頁：${state.message}` : '載入頁面並擷取影片、講義、大綱…'}
        </Text>
      </View>
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
        style={styles.web}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.dark },
  bar: { flexDirection: 'row', alignItems: 'center', height: 48, paddingHorizontal: 6 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center' },
  status: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 14, paddingBottom: 8 },
  statusText: { flex: 1, color: '#9ca3af', fontSize: 12 },
  web: { flex: 1 },
});
