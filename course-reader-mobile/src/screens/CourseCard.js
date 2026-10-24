import React from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { colors } from '../ui';

export default function CourseCard({ course, onPress, footer }) {
  const meta = [course.levels.join('／'), course.topics.slice(0, 2).join('、')].filter(Boolean).join(' · ');
  return (
    <Pressable style={styles.card} onPress={() => onPress(course)} accessibilityRole="button">
      {course.image ? <Image source={{ uri: course.image }} style={styles.img} /> : <View style={[styles.img, styles.noImg]} />}
      <View style={styles.body}>
        <Text style={styles.title} numberOfLines={2}>{course.title}</Text>
        <Text style={styles.source} numberOfLines={1}>
          {course.sourceName}{course.free ? ' · 免費' : ''}{course.unified ? ' · 統一閱讀' : ' · 需登入網頁'}
        </Text>
        {meta ? <Text style={styles.meta} numberOfLines={1}>{meta}</Text> : null}
        {footer}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: { flexDirection: 'row', backgroundColor: colors.card, borderRadius: 10, marginHorizontal: 12, marginBottom: 10, overflow: 'hidden' },
  img: { width: 96, height: 96, backgroundColor: colors.line },
  noImg: { alignItems: 'center', justifyContent: 'center' },
  body: { flex: 1, padding: 10, gap: 3 },
  title: { fontSize: 15, fontWeight: '600', color: colors.text },
  source: { fontSize: 12, color: colors.primary },
  meta: { fontSize: 12, color: colors.muted },
});
