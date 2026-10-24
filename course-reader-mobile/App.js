import 'react-native-url-polyfill/auto';
import React, { useEffect, useState } from 'react';
import { BackHandler, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import CourseDetail from './src/screens/CourseDetail';
import ExtractScreen from './src/screens/ExtractScreen';
import Explore from './src/screens/Explore';
import Library from './src/screens/Library';
import Sites from './src/screens/Sites';
import WebPage from './src/screens/WebPage';
import { colors } from './src/ui';

const TABS = [
  { key: 'explore', label: '探索' },
  { key: 'library', label: '我的課程' },
  { key: 'sites', label: '網站' },
];

function Shell() {
  const [tab, setTab] = useState('explore');
  const [stack, setStack] = useState([]); // 疊在分頁上的畫面：課程詳情、閱讀器、網頁
  const [libraryKey, setLibraryKey] = useState(0);

  const push = (screen) => setStack((s) => [...s, screen]);
  const replaceTop = (screen) => setStack((s) => [...s.slice(0, -1), screen]);
  const pop = () => {
    setStack((s) => s.slice(0, -1));
    setLibraryKey((k) => k + 1); // 回到分頁時重新載入閱讀進度與收藏
  };
  const openCourse = (course) => push({ name: 'detail', course });

  // Android 返回鍵：先關閉最上層畫面，再回到「探索」
  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (stack.length) { pop(); return true; }
      if (tab !== 'explore') { setTab('explore'); return true; }
      return false;
    });
    return () => sub.remove();
  }, [stack.length, tab]);

  const top = stack[stack.length - 1];
  let overlay = null;
  if (top?.name === 'detail') {
    overlay = (
      <CourseDetail
        course={top.course}
        onRead={(course, pages, startIndex) =>
          push({ name: 'extract', url: pages[startIndex].url, course, chapters: pages.length > 1 ? pages : null, chapterIndex: startIndex })
        }
      />
    );
  } else if (top?.name === 'extract') {
    // 所有課程（任何來源、任何入口）都用同一個整理後的統一檢視
    overlay = (
      <ExtractScreen
        key={top.url}
        url={top.url}
        course={top.course}
        chapters={top.chapters}
        chapterIndex={top.chapterIndex}
        onClose={pop}
        onOpenPage={(url) => push({ name: 'extract', url, course: top.course })}
        onOpenChapter={(i) => replaceTop({ ...top, url: top.chapters[i].url, chapterIndex: i })}
        onOpenSource={(url) => push({ name: 'web', title: '原頁面', url })}
      />
    );
  } else if (top?.name === 'web') {
    overlay = <WebPage title={top.title} url={top.url} onClose={pop} />;
  }

  // 閱讀器與網頁自帶工具列；課程詳情則在上方補一條返回列
  const fullScreen = top && top.name !== 'detail';

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <StatusBar style="light" />
      {top ? (
        <View style={{ flex: 1 }}>
          {!fullScreen && (
            <View style={styles.header}>
              <Pressable onPress={pop}><Text style={styles.back}>← 返回</Text></Pressable>
            </View>
          )}
          {overlay}
        </View>
      ) : (
        <View style={{ flex: 1 }}>
          <View style={styles.header}><Text style={styles.headerTitle}>📚 課程閱讀器</Text></View>
          <View style={{ flex: 1 }}>
            {/* 探索頁保持掛載，切換分頁或進入課程後仍保留搜尋條件與結果 */}
            <View style={[styles.fill, tab !== 'explore' && styles.hidden]}><Explore onOpenCourse={openCourse} /></View>
            {tab === 'library' && <Library key={libraryKey} onOpenCourse={openCourse} />}
            {tab === 'sites' && <Sites onOrganize={(url) => push({ name: 'extract', url })} />}
          </View>
          <View style={styles.tabs}>
            {TABS.map((t) => (
              <Pressable key={t.key} style={styles.tab} onPress={() => setTab(t.key)}>
                <Text style={[styles.tabText, tab === t.key && styles.tabActive]}>{t.label}</Text>
              </Pressable>
            ))}
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

export default function App() {
  return <SafeAreaProvider><Shell /></SafeAreaProvider>;
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.dark },
  header: { height: 48, justifyContent: 'center', paddingHorizontal: 14 },
  headerTitle: { color: '#fff', fontSize: 17, fontWeight: '700' },
  back: { color: '#fff', fontSize: 16 },
  fill: { flex: 1 },
  hidden: { display: 'none' },
  tabs: { flexDirection: 'row', backgroundColor: colors.dark, borderTopWidth: 1, borderColor: '#374151' },
  tab: { flex: 1, alignItems: 'center', paddingVertical: 12 },
  tabText: { color: '#9ca3af', fontSize: 14 },
  tabActive: { color: '#fff', fontWeight: '700' },
});
