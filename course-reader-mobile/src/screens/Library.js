import React, { useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { getFavorites, getProgress, removeFavorite, removeProgress } from '../store';
import { colors } from '../ui';
import CourseCard from './CourseCard';

export default function Library({ onOpenCourse }) {
  const [favs, setFavs] = useState([]);
  const [recent, setRecent] = useState([]);
  const [editing, setEditing] = useState(false);

  const load = () => {
    getFavorites().then((f) => setFavs(Object.values(f)));
    getProgress().then((p) => setRecent(Object.values(p).sort((a, b) => b.updatedAt - a.updatedAt)));
  };
  useEffect(load, []);

  const confirmRemove = (title, what, action) =>
    Alert.alert(`移除「${title}」？`, what, [
      { text: '取消', style: 'cancel' },
      { text: '移除', style: 'destructive', onPress: async () => { await action(); load(); } },
    ]);

  const removeBtn = (onPress) => (
    <Pressable onPress={onPress} style={styles.remove} accessibilityRole="button">
      <Text style={styles.removeText}>移除</Text>
    </Pressable>
  );
  const open = (c) => { if (!editing) onOpenCourse(c); };

  return (
    <ScrollView style={styles.root} contentContainerStyle={{ paddingVertical: 12 }}>
      {recent.length + favs.length > 0 ? (
        <View style={styles.editRow}>
          <Pressable onPress={() => setEditing((e) => !e)} accessibilityRole="button">
            <Text style={styles.edit}>{editing ? '完成' : '編輯'}</Text>
          </Pressable>
        </View>
      ) : null}
      <Text style={styles.h}>繼續閱讀</Text>
      {recent.length === 0 ? <Text style={styles.empty}>還沒有閱讀紀錄</Text> : null}
      {recent.map((r) => (
        <CourseCard
          key={r.course.id}
          course={r.course}
          onPress={open}
          footer={
            <View style={styles.footRow}>
              <Text style={styles.progress} numberOfLines={1}>讀到：{r.title}（{r.index + 1}/{r.total}）</Text>
              {editing ? removeBtn(() => confirmRemove(r.course.title, '只會清除閱讀進度，不會影響收藏。', () => removeProgress(r.course.id))) : null}
            </View>
          }
        />
      ))}
      <Text style={styles.h}>收藏</Text>
      {favs.length === 0 ? <Text style={styles.empty}>在課程頁點「☆ 收藏」就會出現在這裡</Text> : null}
      {favs.map((c) => (
        <CourseCard
          key={c.id}
          course={c}
          onPress={open}
          footer={editing ? <View style={styles.footRow}>{removeBtn(() => confirmRemove(c.title, '會從收藏中移除。', () => removeFavorite(c.id)))}</View> : null}
        />
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  editRow: { alignItems: 'flex-end', paddingHorizontal: 14 },
  edit: { color: colors.primary, fontSize: 15, fontWeight: '600', paddingVertical: 4 },
  h: { fontSize: 16, fontWeight: '700', paddingHorizontal: 14, paddingVertical: 8, color: colors.text },
  empty: { color: colors.muted, paddingHorizontal: 14, paddingBottom: 10 },
  footRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  progress: { flex: 1, fontSize: 12, color: colors.muted },
  remove: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 12, backgroundColor: '#fee2e2' },
  removeText: { color: '#b91c1c', fontSize: 13, fontWeight: '600' },
});
