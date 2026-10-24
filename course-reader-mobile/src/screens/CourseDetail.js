import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { fetchOutline, flattenOutline } from '../api/ocw';
import { getFavorites, getProgress, toggleFavorite } from '../store';
import { colors } from '../ui';

const COLLAPSED_CHAPTERS = 4;
const LONG_DESCRIPTION = 160;

export default function CourseDetail({ course, onRead }) {
  const [outline, setOutline] = useState(null); // null=載入中
  const [outlineError, setOutlineError] = useState(false);
  const [fav, setFav] = useState(false);
  const [progress, setProgress] = useState(null);
  const [descOpen, setDescOpen] = useState(false);
  const [chaptersOpen, setChaptersOpen] = useState(false);

  useEffect(() => {
    getFavorites().then((f) => setFav(Boolean(f[course.id])));
    getProgress().then((p) => setProgress(p[course.id] ?? null));
    if (!course.unified) return;
    fetchOutline(course.url)
      .then(setOutline)
      .catch(() => { setOutline([]); setOutlineError(true); });
  }, [course]);

  // 有章節目錄的來源（OCW）要等目錄載入；其他來源直接以課程首頁開始
  const ready = !course.unified || outline !== null;
  const pages = outline ? flattenOutline(outline) : [];
  // 沒抓到章節時退回課程首頁，仍能用同一個檢視閱讀
  const readable = pages.length ? pages : [{ title: course.title, url: course.url, section: course.title }];
  const resumeAt = progress && progress.index < readable.length ? progress.index : 0;
  const longDesc = course.description.length > LONG_DESCRIPTION;
  const tags = [...course.levels, ...course.topics].slice(0, 5);
  const shown = outline ? (chaptersOpen ? outline : outline.slice(0, COLLAPSED_CHAPTERS)) : [];

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ paddingBottom: 32 }}>
      {course.image ? <Image source={{ uri: course.image }} style={styles.hero} resizeMode="cover" /> : null}
      <View style={styles.pad}>
        <Text style={styles.title}>{course.title}</Text>
        <Text style={styles.source}>{course.sourceName}{course.free ? ' · 免費' : ''}</Text>
        {course.instructors.length ? <Text style={styles.meta} numberOfLines={2}>講師：{course.instructors.join('、')}</Text> : null}
        {tags.length ? (
          <View style={styles.tags}>
            {tags.map((t) => <Text key={t} style={styles.tag}>{t}</Text>)}
          </View>
        ) : null}

        <View style={styles.actions}>
          <Pressable style={styles.primary} disabled={!ready} onPress={() => onRead(course, readable, resumeAt)}>
            <Text style={styles.primaryText}>
              {!ready ? '載入章節中…' : progress ? (readable.length > 1 ? `繼續學習（第 ${resumeAt + 1} 章）` : '繼續學習') : '開始學習'}
            </Text>
          </Pressable>
          <Pressable style={styles.secondary} onPress={async () => setFav(await toggleFavorite(course))}>
            <Text style={styles.secondaryText}>{fav ? '★ 已收藏' : '☆ 收藏'}</Text>
          </Pressable>
        </View>

        {!course.unified ? (
          <Text style={styles.note}>此來源需要登入。若進去後沒有內容，請在檢視上方按「原頁面」登入，再回來按「重新整理」，登入狀態會被記住。</Text>
        ) : null}

        {course.description ? (
          <View style={styles.card}>
            <Text style={styles.desc} numberOfLines={descOpen || !longDesc ? undefined : 4}>{course.description}</Text>
            {longDesc ? (
              <Pressable onPress={() => setDescOpen((o) => !o)}>
                <Text style={styles.toggle}>{descOpen ? '收合 ▴' : '展開 ▾'}</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}

        {course.unified ? (
          <View style={styles.card}>
            <View style={styles.cardHead}>
              <Text style={styles.h2}>課程章節</Text>
              {outline ? <Text style={styles.meta}>{outline.length} 章</Text> : null}
            </View>
            {outline === null ? <ActivityIndicator style={{ margin: 8 }} /> : null}
            {outlineError ? <Text style={styles.note}>章節載入失敗，仍可從課程首頁開始學習。</Text> : null}
            {outline && outline.length === 0 && !outlineError ? (
              <Text style={styles.note}>這門課沒有章節目錄，將直接開啟課程首頁。</Text>
            ) : null}
            {shown.map((s, i) => (
              <Pressable
                key={s.key}
                style={styles.chapterRow}
                onPress={() => onRead(course, readable, Math.max(0, readable.findIndex((p) => p.section === s.title)))}
              >
                <Text style={styles.chapterNo}>{i + 1}</Text>
                <Text style={styles.chapterTitle} numberOfLines={1}>{s.title}</Text>
                {s.items.length ? <Text style={styles.meta}>{s.items.length} 個單元</Text> : null}
              </Pressable>
            ))}
            {outline && outline.length > COLLAPSED_CHAPTERS ? (
              <Pressable onPress={() => setChaptersOpen((o) => !o)}>
                <Text style={styles.toggle}>{chaptersOpen ? '收合 ▴' : `顯示全部 ${outline.length} 章 ▾`}</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  hero: { width: '100%', height: 150, backgroundColor: colors.line },
  pad: { padding: 14, gap: 6 },
  title: { fontSize: 19, fontWeight: '700', color: colors.text },
  source: { color: colors.primary, fontSize: 13 },
  meta: { color: colors.muted, fontSize: 12 },
  tags: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 2 },
  tag: { fontSize: 11, color: colors.text, backgroundColor: colors.line, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10, overflow: 'hidden' },
  actions: { flexDirection: 'row', gap: 10, marginTop: 8 },
  primary: { flex: 1, backgroundColor: colors.primary, borderRadius: 8, padding: 12, alignItems: 'center' },
  primaryText: { color: '#fff', fontWeight: '600' },
  secondary: { borderWidth: 1, borderColor: colors.primary, borderRadius: 8, padding: 12, alignItems: 'center' },
  secondaryText: { color: colors.primary, fontWeight: '600' },
  note: { color: colors.muted, fontSize: 12, marginTop: 4 },
  card: { backgroundColor: colors.card, borderRadius: 10, padding: 12, marginTop: 8 },
  cardHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  desc: { lineHeight: 21, color: colors.text, fontSize: 14 },
  toggle: { color: colors.primary, fontSize: 13, paddingTop: 8, textAlign: 'center' },
  h2: { fontSize: 15, fontWeight: '700', color: colors.text },
  chapterRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 9, borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  chapterNo: { width: 20, textAlign: 'center', color: colors.primary, fontSize: 13 },
  chapterTitle: { flex: 1, fontSize: 14, color: colors.text },
});
