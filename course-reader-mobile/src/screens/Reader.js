import React, { useEffect, useRef, useState } from 'react';
import { Linking, Modal, Pressable, FlatList, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { sameUrl } from '../api/ocw';
import { READER_JS } from '../reader/inject';
import { saveProgress } from '../store';
import { Btn, colors } from '../ui';

// 統一閱讀介面：課程頁面內容、導覽列、章節目錄與進度都由本 App 提供
export default function Reader({ course, pages, startIndex = 0, onClose, onOrganize }) {
  const [index, setIndex] = useState(startIndex);
  const [toc, setToc] = useState(false);
  const web = useRef(null);
  const page = pages[index];
  const coursePath = new URL(course.url).pathname.replace(/\/+$/, '');

  useEffect(() => {
    saveProgress(course, { index, title: page.title, url: page.url, total: pages.length });
  }, [index]);

  const go = (i) => i >= 0 && i < pages.length && setIndex(i);

  const onShouldStart = (req) => {
    if (req.isTopFrame === false) return true; // 內嵌的影片播放器
    if (sameUrl(req.url, page.url) || req.url === 'about:blank') return true;
    const j = pages.findIndex((p) => sameUrl(p.url, req.url));
    if (j >= 0) { setIndex(j); return false; }
    try {
      const u = new URL(req.url);
      if (u.hostname === new URL(course.url).hostname && u.pathname.startsWith(`${coursePath}/`)) return true;
    } catch {}
    Linking.openURL(req.url).catch(() => {});
    return false;
  };

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.bar}>
        <Btn label="✕" onPress={onClose} />
        <Text style={styles.title} numberOfLines={1}>{page.title}</Text>
        {onOrganize ? <Btn label="整理" onPress={() => onOrganize(page.url)} /> : null}
        <Btn label="目錄" onPress={() => setToc(true)} />
      </View>
      <View style={styles.progress}>
        <View style={[styles.progressFill, { width: `${((index + 1) / pages.length) * 100}%` }]} />
      </View>
      <WebView
        key={page.url}
        ref={web}
        source={{ uri: page.url }}
        injectedJavaScript={READER_JS}
        onShouldStartLoadWithRequest={onShouldStart}
        domStorageEnabled
        javaScriptEnabled
        allowsInlineMediaPlayback
        mediaPlaybackRequiresUserAction={false}
        allowsFullscreenVideo
        setSupportMultipleWindows={false}
        style={styles.web}
      />
      <View style={styles.bar}>
        <Btn label="← 上一頁" disabled={index === 0} onPress={() => go(index - 1)} />
        <Text style={styles.count}>{index + 1} / {pages.length}</Text>
        <Btn label="下一頁 →" disabled={index === pages.length - 1} onPress={() => go(index + 1)} />
      </View>

      <Modal visible={toc} animationType="slide" onRequestClose={() => setToc(false)}>
        <SafeAreaView style={styles.toc}>
          <View style={styles.tocHead}>
            <Text style={styles.tocTitle} numberOfLines={1}>{course.title}</Text>
            <Btn label="關閉" dark onPress={() => setToc(false)} />
          </View>
          <FlatList
            data={pages}
            keyExtractor={(p) => p.url}
            initialScrollIndex={Math.max(0, Math.min(index, pages.length - 1))}
            getItemLayout={(_, i) => ({ length: 52, offset: 52 * i, index: i })}
            renderItem={({ item, index: i }) => (
              <Pressable style={[styles.tocRow, i === index && styles.tocRowActive]} onPress={() => { go(i); setToc(false); }}>
                <Text style={styles.tocSection} numberOfLines={1}>{item.section}</Text>
                <Text style={styles.tocText} numberOfLines={1}>{item.title}</Text>
              </Pressable>
            )}
          />
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.dark },
  bar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', height: 48, paddingHorizontal: 6 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
  count: { color: '#9ca3af' },
  progress: { height: 3, backgroundColor: '#374151' },
  progressFill: { height: 3, backgroundColor: colors.primary },
  web: { flex: 1 },
  toc: { flex: 1, backgroundColor: '#fff' },
  tocHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 12 },
  tocTitle: { flex: 1, fontSize: 16, fontWeight: '700' },
  tocRow: { height: 52, paddingHorizontal: 16, justifyContent: 'center', borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#d1d5db' },
  tocRowActive: { backgroundColor: colors.primarySoft },
  tocSection: { fontSize: 11, color: colors.muted },
  tocText: { fontSize: 15 },
});
