import React, { useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text } from 'react-native';
import { getFavorites, getProgress } from '../store';
import { colors } from '../ui';
import CourseCard from './CourseCard';

export default function Library({ onOpenCourse }) {
  const [favs, setFavs] = useState([]);
  const [recent, setRecent] = useState([]);

  useEffect(() => {
    getFavorites().then((f) => setFavs(Object.values(f)));
    getProgress().then((p) => setRecent(Object.values(p).sort((a, b) => b.updatedAt - a.updatedAt)));
  }, []);

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ paddingVertical: 12 }}>
      <Text style={styles.h}>繼續閱讀</Text>
      {recent.length === 0 ? <Text style={styles.empty}>還沒有閱讀紀錄</Text> : null}
      {recent.map((r) => (
        <CourseCard
          key={r.course.id}
          course={r.course}
          onPress={onOpenCourse}
          footer={<Text style={styles.progress} numberOfLines={1}>讀到：{r.title}（{r.index + 1}/{r.total}）</Text>}
        />
      ))}
      <Text style={styles.h}>收藏</Text>
      {favs.length === 0 ? <Text style={styles.empty}>在課程頁點「☆ 收藏」就會出現在這裡</Text> : null}
      {favs.map((c) => <CourseCard key={c.id} course={c} onPress={onOpenCourse} />)}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  h: { fontSize: 16, fontWeight: '700', paddingHorizontal: 14, paddingVertical: 8, color: colors.text },
  empty: { color: colors.muted, paddingHorizontal: 14, paddingBottom: 10 },
  progress: { fontSize: 12, color: colors.muted },
});
