import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { fetchOutline, flattenOutline } from '../api/ocw';
import { getFavorites, getProgress, toggleFavorite } from '../store';
import { colors } from '../ui';

export default function CourseDetail({ course, onRead }) {
  const [outline, setOutline] = useState(null); // null=載入中
  const [outlineError, setOutlineError] = useState(false);
  const [fav, setFav] = useState(false);
  const [progress, setProgress] = useState(null);

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
  // 沒抓到章節時退回課程首頁，仍能在統一介面內閱讀
  const readable = pages.length ? pages : [{ title: course.title, url: course.url, section: course.title }];
  const resumeAt = progress && progress.index < readable.length ? progress.index : 0;

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ paddingBottom: 40 }}>
      {course.image ? <Image source={{ uri: course.image }} style={styles.hero} /> : null}
      <View style={styles.pad}>
        <Text style={styles.title}>{course.title}</Text>
        <Text style={styles.source}>
          {course.sourceName}{course.free ? ' · 免費' : ''}
        </Text>
        {course.instructors.length ? <Text style={styles.meta}>講師：{course.instructors.join('、')}</Text> : null}
        {course.levels.length ? <Text style={styles.meta}>程度：{course.levels.join('／')}</Text> : null}
        {course.topics.length ? <Text style={styles.meta}>主題：{course.topics.join('、')}</Text> : null}
        {course.features.length ? <Text style={styles.meta}>內容：{course.features.join('、')}</Text> : null}

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

        {course.description ? <Text style={styles.desc}>{course.description}</Text> : null}

        {course.unified ? (
          <View style={{ marginTop: 16 }}>
            <Text style={styles.h2}>課程章節</Text>
            {outline === null ? <ActivityIndicator /> : null}
            {outlineError ? <Text style={styles.note}>章節載入失敗，仍可從課程首頁開始閱讀。</Text> : null}
            {outline && outline.length === 0 && !outlineError ? (
              <Text style={styles.note}>這門課沒有章節目錄，將直接開啟課程首頁。</Text>
            ) : null}
            {outline?.map((s) => (
              <View key={s.key} style={{ marginBottom: 6 }}>
                <Pressable onPress={() => onRead(course, readable, Math.max(0, readable.findIndex((p) => p.section === s.title)))}>
                  <Text style={styles.section}>{s.title}</Text>
                </Pressable>
                <Text style={styles.meta}>{s.items.length ? `${s.items.length} 個單元` : '單頁'}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  hero: { width: '100%', height: 190, backgroundColor: colors.line },
  pad: { padding: 16, gap: 4 },
  title: { fontSize: 20, fontWeight: '700', color: colors.text },
  source: { color: colors.primary, marginBottom: 4 },
  meta: { color: colors.muted, fontSize: 13 },
  actions: { flexDirection: 'row', gap: 10, marginTop: 12 },
  primary: { flex: 1, backgroundColor: colors.primary, borderRadius: 8, padding: 12, alignItems: 'center' },
  primaryText: { color: '#fff', fontWeight: '600' },
  secondary: { borderWidth: 1, borderColor: colors.primary, borderRadius: 8, padding: 12, alignItems: 'center' },
  secondaryText: { color: colors.primary, fontWeight: '600' },
  note: { color: colors.muted, fontSize: 12, marginTop: 8 },
  desc: { marginTop: 14, lineHeight: 22, color: colors.text },
  h2: { fontSize: 16, fontWeight: '700', marginBottom: 6 },
  section: { fontSize: 15, fontWeight: '600', color: colors.text },
});
