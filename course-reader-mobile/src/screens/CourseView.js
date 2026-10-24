import React, { useMemo, useState } from 'react';
import { FlatList, Linking, Modal, Pressable, ScrollView, Share, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { extOf, playerHtml, tabsOf, viewerUrl, youtubeWatchUrl } from '../content/build';
import { resolveFileUrl } from '../content/resolve';
import { Btn, Chip, colors } from '../ui';

const FILE_ICON = { pdf: '📄', ppt: '📊', pptx: '📊', doc: '📝', docx: '📝', zip: '🗜', xls: '📈', xlsx: '📈' };
const IN_APP_VIEW = /^(pdf|pptx?|docx?|xlsx?)$/;

function VideoPlayer({ video, origin }) {
  // 以來源網站的網域作為嵌入來源，與原頁面相同，YouTube 的嵌入限制才會一致
  return (
    <View style={styles.player}>
      <WebView
        key={video.id}
        source={{ html: playerHtml(video), baseUrl: origin }}
        allowsInlineMediaPlayback
        allowsFullscreenVideo
        mediaPlaybackRequiresUserAction={false}
        javascriptEnabled
        domStorageEnabled
        setSupportMultipleWindows={false}
        style={{ flex: 1, backgroundColor: '#000' }}
      />
    </View>
  );
}

function FileViewer({ file, onClose }) {
  return (
    <Modal visible animationType="slide" onRequestClose={onClose}>
      <SafeAreaView style={styles.viewer}>
        <View style={styles.viewerBar}>
          <Btn label="← 返回" onPress={onClose} />
          <Text style={styles.viewerTitle} numberOfLines={1}>{file.title}</Text>
          <Btn label="下載" onPress={() => Linking.openURL(file.url).catch(() => {})} />
        </View>
        <WebView source={{ uri: viewerUrl(file.url) }} javaScriptEnabled domStorageEnabled style={{ flex: 1 }} />
      </SafeAreaView>
    </Modal>
  );
}

// 可點的一列：標題在左，右邊有「›」表示點了會進去
function Row({ title, subtitle, onPress, active }) {
  return (
    <Pressable style={[styles.row, active && styles.rowActive]} onPress={onPress}>
      <View style={{ flex: 1 }}>
        <Text style={styles.rowTitle} numberOfLines={2}>{active ? '▶ ' : ''}{title}</Text>
        {subtitle ? <Text style={styles.rowMeta} numberOfLines={1}>{subtitle}</Text> : null}
      </View>
      <Text style={styles.chevron}>›</Text>
    </Pressable>
  );
}

// 統一的課程檢視：不論來源網站，都用相同的分類與版面呈現
// - 課程頁／單元頁：依內容分成影片、講義、作業…等分頁（只有一個分頁時不顯示分頁列）
// - 列表型頁面（只有一串可選的項目，如網站首頁、課程列表）：直接顯示清單，不放分頁

export default function CourseView({ content, chapters, chapterIndex, onOpenPage, onOpenChapter, onOpenSource, onRefresh, onClose, onHome }) {
  const hasChapters = Boolean(chapters && chapters.length > 1);
  const listMode = !hasChapters && content.videos.length === 0 && content.materials.length === 0 && content.related.length >= 3;
  const tabs = useMemo(() => {
    const t = tabsOf(content, { hideRelated: hasChapters });
    return hasChapters ? [{ key: 'chapters', label: '章節', count: chapters.length }, ...t] : t;
  }, [content, hasChapters, chapters]);
  // 有影片就先看影片；否則第一個有內容的分頁；這頁沒內容但有章節時顯示章節
  // 優先順序：影片 → 課程首頁（尚未進入任何章節）顯示章節清單 → 第一個有內容的分頁
  const [tab, setTab] = useState(() => {
    const t = tabs.find((x) => x.key === 'videos')
      ?? (hasChapters && typeof chapterIndex !== 'number' ? tabs.find((x) => x.key === 'chapters') : null)
      ?? tabs.find((x) => x.key !== 'chapters')
      ?? tabs[0];
    return t?.key ?? 'overview';
  });
  const [videoId, setVideoId] = useState(content.videos[0]?.id ?? null);
  const [file, setFile] = useState(null);
  const [openSecs, setOpenSecs] = useState(() => new Set([0])); // 說明文字的區塊，預設只展開第一個
  const toggleSec = (i) => setOpenSecs((o) => { const n = new Set(o); if (n.has(i)) n.delete(i); else n.add(i); return n; });
  const origin = useMemo(() => { try { return new URL(content.url).origin; } catch { return 'https://localhost'; } }, [content.url]);
  const video = content.videos.find((v) => v.id === videoId) ?? content.videos[0];

  const openFile = async (m) => {
    let target = m;
    if (m.resolve) {
      const url = await resolveFileUrl(m.url).catch(() => null);
      if (!url) return Linking.openURL(m.url).catch(() => {});
      target = { ...m, url, ext: extOf(url) || m.ext };
    }
    return IN_APP_VIEW.test(target.ext) ? setFile(target) : Linking.openURL(target.url).catch(() => {});
  };
  const materials = (cat) => content.materials.filter((m) => m.category === cat);

  const renderMaterial = (m) => (
    <Pressable key={m.id} style={styles.row} onPress={() => openFile(m)}>
      <Text style={styles.icon}>{FILE_ICON[m.ext] ?? '📎'}</Text>
      <View style={{ flex: 1 }}>
        <Text style={styles.rowTitle} numberOfLines={2}>{m.title}</Text>
        <Text style={styles.rowMeta} numberOfLines={1}>{[m.ext.toUpperCase(), m.section].filter(Boolean).join(' · ')}</Text>
      </View>
    </Pressable>
  );

  let body = null;
  const sectionsDiffer = hasChapters && new Set(chapters.map((c) => c.section)).size > 1;
  if (listMode) {
    body = (
      <ScrollView contentContainerStyle={styles.listPad}>
        <Text style={styles.hint}>請選擇要查看的項目（共 {content.related.length} 項）</Text>
        {content.related.map((r) => <Row key={r.id} title={r.title} onPress={() => onOpenPage(r.url)} />)}
      </ScrollView>
    );
  } else if (tab === 'chapters') {
    body = (
      <FlatList
        data={chapters}
        keyExtractor={(c) => c.url}
        initialScrollIndex={Math.max(0, Math.min(chapterIndex ?? 0, chapters.length - 1))}
        getItemLayout={(_, i) => ({ length: 56, offset: 56 * i, index: i })}
        renderItem={({ item, index }) => (
          <Pressable style={[styles.chapter, index === chapterIndex && styles.rowActive]} onPress={() => onOpenChapter(index, chapters)}>
            <View style={{ flex: 1 }}>
              {sectionsDiffer ? <Text style={styles.rowMeta} numberOfLines={1}>{item.section}</Text> : null}
              <Text style={styles.rowTitle} numberOfLines={1}>{index === chapterIndex ? '▶ ' : ''}{item.title}</Text>
            </View>
            <Text style={styles.chevron}>›</Text>
          </Pressable>
        )}
      />
    );
  } else if (tab === 'overview') {
    const secs = content.overviewSections ?? [];
    const grouped = secs.length >= 2;
    body = (
      <ScrollView contentContainerStyle={styles.pad}>
        {grouped ? (
          secs.map((sec, i) => (
            <View key={sec.title} style={styles.acc}>
              <Pressable style={styles.accHead} onPress={() => toggleSec(i)}>
                <Text style={styles.accTitle}>{sec.title}</Text>
                <Text style={styles.chev}>{openSecs.has(i) ? '▴' : '▾'}</Text>
              </Pressable>
              {openSecs.has(i) ? sec.paragraphs.map((p, j) => <Text key={j} style={styles.para}>{p}</Text>) : null}
            </View>
          ))
        ) : (
          <>
            {content.overview.map((p, i) => <Text key={i} style={styles.para}>{p}</Text>)}
          </>
        )}
        {materials('overview').length ? <Text style={[styles.h2, { marginTop: content.overview.length ? 16 : 0 }]}>大綱與課表檔案</Text> : null}
        {materials('overview').map(renderMaterial)}
      </ScrollView>
    );
  } else if (tab === 'videos') {
    body = (
      <View style={{ flex: 1 }}>
        {video ? <VideoPlayer video={video} origin={origin} /> : null}
        {video && video.kind !== 'mp4' ? (
          <Pressable onPress={() => Linking.openURL(youtubeWatchUrl(video)).catch(() => {})}>
            <Text style={styles.fallback}>影片無法播放？在 YouTube 開啟</Text>
          </Pressable>
        ) : null}
        <FlatList
          data={content.videos}
          keyExtractor={(v) => v.id}
          renderItem={({ item, index }) => (
            <Pressable style={[styles.row, item.id === video?.id && styles.rowActive]} onPress={() => setVideoId(item.id)}>
              <Text style={styles.icon}>{item.id === video?.id ? '▶' : String(index + 1)}</Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.rowTitle} numberOfLines={2}>{item.title}</Text>
                {item.section ? <Text style={styles.rowMeta} numberOfLines={1}>{item.section}</Text> : null}
              </View>
            </Pressable>
          )}
        />
      </View>
    );
  } else if (tab === 'related') {
    body = (
      <ScrollView contentContainerStyle={styles.listPad}>
        <Text style={styles.hint}>點選項目查看內容</Text>
        {content.related.map((r) => <Row key={r.id} title={r.title} onPress={() => onOpenPage(r.url)} />)}
      </ScrollView>
    );
  } else {
    body = <ScrollView contentContainerStyle={styles.pad}>{materials(tab).map(renderMaterial)}</ScrollView>;
  }

  return (
    <SafeAreaView style={styles.root} edges={[]}>
      <View style={styles.head}>
        <Btn label="← 返回" onPress={onClose} />
        {onHome ? <Btn label="首頁" onPress={onHome} /> : null}
        <Text style={styles.title} numberOfLines={1}>{(hasChapters && chapters[chapterIndex]?.title) || content.title}</Text>
        <Btn label="原頁面" onPress={() => onOpenSource(content.url)} />
      </View>

      {tabs.length === 0 && !listMode ? (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>這個頁面沒有找到影片、講義或大綱。</Text>
          <Text style={styles.note}>如果需要登入，請按「原頁面」登入後回來重新整理；也可以分享診斷資訊給開發者。</Text>
          {onRefresh ? <Btn label="重新整理" dark onPress={onRefresh} /> : null}
        </View>
      ) : (
        <>
          {tabs.length >= 2 && !listMode ? (
            <View style={styles.tabs}>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 10 }}>
                {tabs.map((t) => (
                  <Chip key={t.key} label={t.key === 'overview' ? t.label : `${t.label} ${t.count}`} active={tab === t.key} onPress={() => setTab(t.key)} />
                ))}
              </ScrollView>
            </View>
          ) : null}
          <View style={{ flex: 1 }}>{body}</View>
        </>
      )}

      {hasChapters && typeof chapterIndex === 'number' ? (
        <View style={styles.chapterNav}>
          <Btn label="← 上一章" dark disabled={chapterIndex === 0} onPress={() => onOpenChapter(chapterIndex - 1, chapters)} />
          <Text style={styles.source}>{chapterIndex + 1} / {chapters.length}</Text>
          <Btn label="下一章 →" dark disabled={chapterIndex === chapters.length - 1} onPress={() => onOpenChapter(chapterIndex + 1, chapters)} />
        </View>
      ) : null}
      <View style={styles.foot}>
        <Text style={styles.source} numberOfLines={1}>內容來源：{content.source}</Text>
        <Pressable onPress={() => Share.share({ message: JSON.stringify(content.diagnostic ?? { url: content.url, counts: { videos: content.videos.length, materials: content.materials.length, related: content.related.length } }, null, 1) })}>
          <Text style={styles.share}>分享診斷資訊</Text>
        </Pressable>
      </View>
      {file ? <FileViewer file={file} onClose={() => setFile(null)} /> : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  head: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.dark, height: 48, paddingHorizontal: 6 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
  tabs: { paddingVertical: 8, backgroundColor: colors.bg },
  pad: { padding: 14, paddingBottom: 40 },
  h2: { fontSize: 15, fontWeight: '700', marginBottom: 6, color: colors.text },
  outline: { fontSize: 14, color: colors.text, paddingVertical: 2 },
  para: { fontSize: 14, lineHeight: 22, color: colors.text, marginBottom: 8 },
  note: { fontSize: 12, color: colors.muted, marginBottom: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: colors.card, paddingHorizontal: 14, paddingVertical: 11, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  chevron: { fontSize: 22, color: colors.muted, paddingLeft: 8 },
  hint: { fontSize: 13, color: colors.muted, paddingHorizontal: 14, paddingTop: 12, paddingBottom: 8 },
  listPad: { paddingBottom: 40 },
  rowActive: { backgroundColor: colors.primarySoft },
  icon: { width: 24, textAlign: 'center', fontSize: 16, color: colors.primary },
  rowTitle: { fontSize: 14, color: colors.text },
  rowMeta: { fontSize: 11, color: colors.muted, marginTop: 2 },
  player: { width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000' },
  fallback: { color: colors.primary, fontSize: 12, textAlign: 'center', padding: 8 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyText: { fontSize: 15, color: colors.text, marginBottom: 8, textAlign: 'center' },
  acc: { backgroundColor: colors.card, borderRadius: 8, marginBottom: 8, paddingHorizontal: 12 },
  accHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12 },
  accTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  chev: { color: colors.muted, fontSize: 14 },
  chapter: { height: 56, flexDirection: 'row', alignItems: 'center', paddingHorizontal: 14, backgroundColor: colors.card, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  chapterNav: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: colors.card, borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  foot: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 14, paddingVertical: 8, backgroundColor: colors.card, borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  source: { flex: 1, fontSize: 11, color: colors.muted },
  share: { fontSize: 11, color: colors.primary },
  viewer: { flex: 1, backgroundColor: colors.dark },
  viewerBar: { flexDirection: 'row', alignItems: 'center', height: 48, paddingHorizontal: 6 },
  viewerTitle: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
});
