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
          <Btn label="✕" onPress={onClose} />
          <Text style={styles.viewerTitle} numberOfLines={1}>{file.title}</Text>
          <Btn label="下載" onPress={() => Linking.openURL(file.url).catch(() => {})} />
        </View>
        <WebView source={{ uri: viewerUrl(file.url) }} javaScriptEnabled domStorageEnabled style={{ flex: 1 }} />
      </SafeAreaView>
    </Modal>
  );
}

// 統一的課程檢視：不論來源網站，都用相同的分類與版面呈現
export default function CourseView({ content, chapters, chapterIndex, onOpenPage, onOpenChapter, onOpenSource, onRefresh, onClose }) {
  const hasChapters = Boolean(chapters && chapters.length > 1);
  const tabs = useMemo(() => {
    const t = tabsOf(content, { hideRelated: hasChapters });
    return hasChapters ? [{ key: 'chapters', label: '章節', count: chapters.length }, ...t] : t;
  }, [content, hasChapters, chapters]);
  // 有影片就先看影片；否則第一個有內容的分頁；這頁沒內容但有章節時顯示章節
  const [tab, setTab] = useState(() => (tabs.find((t) => t.key === 'videos') ?? tabs.find((t) => t.key !== 'chapters') ?? tabs[0])?.key ?? 'overview');
  const [videoId, setVideoId] = useState(content.videos[0]?.id ?? null);
  const [file, setFile] = useState(null);
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
  if (tab === 'chapters') {
    body = (
      <FlatList
        data={chapters}
        keyExtractor={(c) => c.url}
        initialScrollIndex={Math.max(0, Math.min(chapterIndex ?? 0, chapters.length - 1))}
        getItemLayout={(_, i) => ({ length: 56, offset: 56 * i, index: i })}
        renderItem={({ item, index }) => (
          <Pressable style={[styles.chapter, index === chapterIndex && styles.rowActive]} onPress={() => onOpenChapter(index)}>
            <Text style={styles.rowMeta} numberOfLines={1}>{item.section}</Text>
            <Text style={styles.rowTitle} numberOfLines={1}>{index === chapterIndex ? '▶ ' : ''}{item.title}</Text>
          </Pressable>
        )}
      />
    );
  } else if (tab === 'overview') {
    body = (
      <ScrollView contentContainerStyle={styles.pad}>
        {content.outline.length ? <Text style={styles.h2}>內容結構</Text> : null}
        {content.outline.map((h, i) => (
          <Text key={i} style={[styles.outline, { paddingLeft: Math.max(0, h.level - 1) * 12 }]}>{h.text}</Text>
        ))}
        {content.overview.length ? <Text style={[styles.h2, { marginTop: 16 }]}>課程說明</Text> : null}
        {content.overview.map((p, i) => <Text key={i} style={styles.para}>{p}</Text>)}
        {materials('overview').length ? <Text style={[styles.h2, { marginTop: 16 }]}>大綱與課表檔案</Text> : null}
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
      <ScrollView contentContainerStyle={styles.pad}>
        <Text style={styles.note}>點選後會把該頁面也整理成統一格式。</Text>
        {content.related.map((r) => (
          <Pressable key={r.id} style={styles.row} onPress={() => onOpenPage(r.url)}>
            <Text style={styles.icon}>🔗</Text>
            <Text style={[styles.rowTitle, { flex: 1 }]} numberOfLines={2}>{r.title}</Text>
          </Pressable>
        ))}
      </ScrollView>
    );
  } else {
    body = <ScrollView contentContainerStyle={styles.pad}>{materials(tab).map(renderMaterial)}</ScrollView>;
  }

  return (
    <SafeAreaView style={styles.root} edges={[]}>
      <View style={styles.head}>
        <Btn label="✕" onPress={onClose} />
        <Text style={styles.title} numberOfLines={1}>{(hasChapters && chapters[chapterIndex]?.title) || content.title}</Text>
        <Btn label="原頁面" onPress={() => onOpenSource(content.url)} />
      </View>

      {tabs.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>這個頁面沒有找到影片、講義或大綱。</Text>
          <Text style={styles.note}>如果需要登入，請按「原頁面」登入後回來重新整理；也可以分享診斷資訊給開發者。</Text>
          {onRefresh ? <Btn label="重新整理" dark onPress={onRefresh} /> : null}
        </View>
      ) : (
        <>
          <View style={styles.tabs}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 10 }}>
              {tabs.map((t) => (
                <Chip key={t.key} label={t.key === 'overview' ? t.label : `${t.label} ${t.count}`} active={tab === t.key} onPress={() => setTab(t.key)} />
              ))}
            </ScrollView>
          </View>
          <View style={{ flex: 1 }}>{body}</View>
        </>
      )}

      {hasChapters && typeof chapterIndex === 'number' ? (
        <View style={styles.chapterNav}>
          <Btn label="← 上一章" dark disabled={chapterIndex === 0} onPress={() => onOpenChapter(chapterIndex - 1)} />
          <Text style={styles.source}>{chapterIndex + 1} / {chapters.length}</Text>
          <Btn label="下一章 →" dark disabled={chapterIndex === chapters.length - 1} onPress={() => onOpenChapter(chapterIndex + 1)} />
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
  rowActive: { backgroundColor: colors.primarySoft },
  icon: { width: 24, textAlign: 'center', fontSize: 16, color: colors.primary },
  rowTitle: { fontSize: 14, color: colors.text },
  rowMeta: { fontSize: 11, color: colors.muted, marginTop: 2 },
  player: { width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000' },
  fallback: { color: colors.primary, fontSize: 12, textAlign: 'center', padding: 8 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyText: { fontSize: 15, color: colors.text, marginBottom: 8, textAlign: 'center' },
  chapter: { height: 56, justifyContent: 'center', paddingHorizontal: 14, backgroundColor: colors.card, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  chapterNav: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: colors.card, borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  foot: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 14, paddingVertical: 8, backgroundColor: colors.card, borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  source: { flex: 1, fontSize: 11, color: colors.muted },
  share: { fontSize: 11, color: colors.primary },
  viewer: { flex: 1, backgroundColor: colors.dark },
  viewerBar: { flexDirection: 'row', alignItems: 'center', height: 48, paddingHorizontal: 6 },
  viewerTitle: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
});
