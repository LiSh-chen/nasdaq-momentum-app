import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { fetchOutline, flattenOutline } from '../api/ocw';
import { getFavorites, getProgress, toggleFavorite } from '../store';
import { colors } from '../ui';

export default function CourseDetail({ course, onRead, onOpenWeb }) {
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
          {course.unified ? (
            <Pressable style={styles.primary} disabled={outline === null} onPress={() => onRead(course, readable, resumeAt)}>
              <Text style={styles.primaryText}>
                {outline === null ? '載入章節中…' : progress ? `繼續閱讀（第 ${resumeAt + 1} 頁）` : '開始閱讀'}
              </Text>
            </Pressable>
          ) : (
            <Pressable style={styles.primary} onPress={() => onOpenWeb(course)}>
              <Text style={styles.primaryText}>在 App 內開啟（需登入）</Text>
            </Pressable>
          )}
          <Pressable style={styles.secondary} onPress={async () => setFav(await toggleFavorite(course))}>
            <Text style={styles.secondaryText}>{fav ? '★ 已收藏' : '☆ 收藏'}</Text>
          </Pressable>
        </View>

        {!course.unified ? (
          <Text style={styles.note}>此來源的課程內容需登入且受平台條款保護，因此以 App 內網頁開啟，登入狀態會被記住。</Text>
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
