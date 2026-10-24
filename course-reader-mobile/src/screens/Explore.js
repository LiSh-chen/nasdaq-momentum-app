import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { LEVEL_NAMES, SOURCE_NAMES, searchCourses } from '../api/mitLearn';
import { Chip, colors } from '../ui';
import CourseCard from './CourseCard';

const EMPTY = { items: [], count: 0, facets: {}, loading: false, error: null, hasMore: false };

// 已選的項目即使不在這次的篩選統計裡，也要留在列上讓使用者能取消
const withSelected = (list = [], selected) =>
  selected && !list.some((f) => f.key === selected) ? [{ key: selected, count: 0 }, ...list] : list;

export default function Explore({ onOpenCourse }) {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({ offeredBy: null, level: null, topic: null, hasVideos: false, sort: null });
  const [state, setState] = useState({ ...EMPTY, loading: true });
  const reqId = useRef(0);

  const run = useCallback(
    async (offset, append) => {
      const id = ++reqId.current;
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const r = await searchCourses({ q: query, ...filters, offset });
        if (id !== reqId.current) return; // 已有較新的搜尋，丟棄舊結果
        setState((s) => ({
          items: append ? [...s.items, ...r.items] : r.items,
          count: r.count,
          facets: append ? s.facets : r.facets,
          hasMore: r.hasMore,
          loading: false,
          error: null,
        }));
      } catch {
        if (id !== reqId.current) return;
        setState((s) => ({ ...s, loading: false, error: '載入失敗，請檢查網路後下拉或再試一次' }));
      }
    },
    [query, filters]
  );

  useEffect(() => {
    const t = setTimeout(() => run(0, false), query ? 400 : 0);
    return () => clearTimeout(t);
  }, [run]);

  const set = (key, value) => setFilters((f) => ({ ...f, [key]: f[key] === value ? null : value }));
  const { facets } = state;

  const header = (
    <View>
      <View style={styles.searchBox}>
        <TextInput
          style={styles.input}
          placeholder="搜尋課程、主題、講師…"
          value={query}
          onChangeText={setQuery}
          returnKeyType="search"
          clearButtonMode="while-editing"
        />
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        <Chip label="全部來源" active={!filters.offeredBy} onPress={() => setFilters((f) => ({ ...f, offeredBy: null }))} />
        {withSelected(facets.offered_by, filters.offeredBy).map((f) => (
          <Chip key={f.key} label={SOURCE_NAMES[f.key] ?? f.key.toUpperCase()} active={filters.offeredBy === f.key}
            onPress={() => set('offeredBy', f.key)} />
        ))}
      </ScrollView>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        <Chip label="有講課影片" active={filters.hasVideos} onPress={() => setFilters((f) => ({ ...f, hasVideos: !f.hasVideos }))} />
        <Chip label="熱門排序" active={filters.sort === '-views'} onPress={() => set('sort', '-views')} />
        {withSelected(facets.level, filters.level).map((f) => (
          <Chip key={f.key} label={LEVEL_NAMES[f.key] ?? f.key} active={filters.level === f.key} onPress={() => set('level', f.key)} />
        ))}
      </ScrollView>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        {withSelected(facets.topic?.slice(0, 14), filters.topic).map((f) => (
          <Chip key={f.key} label={f.key} active={filters.topic === f.key} onPress={() => set('topic', f.key)} />
        ))}
      </ScrollView>
      <Text style={styles.count}>{state.loading && !state.items.length ? '搜尋中…' : `共 ${state.count} 門課程`}</Text>
      {state.error ? <Text style={styles.error}>{state.error}</Text> : null}
    </View>
  );

  return (
    <FlatList
      style={styles.list}
      data={state.items}
      keyExtractor={(c) => String(c.id)}
      renderItem={({ item }) => <CourseCard course={item} onPress={onOpenCourse} />}
      ListHeaderComponent={header}
      ListFooterComponent={state.loading ? <ActivityIndicator style={{ margin: 16 }} /> : null}
      ListEmptyComponent={!state.loading && !state.error ? <Text style={styles.empty}>找不到符合的課程</Text> : null}
      onEndReached={() => state.hasMore && !state.loading && run(state.items.length, true)}
      onEndReachedThreshold={0.6}
      keyboardShouldPersistTaps="handled"
    />
  );
}

const styles = StyleSheet.create({
  list: { flex: 1, backgroundColor: colors.bg },
  searchBox: { padding: 12, paddingBottom: 4 },
  input: { backgroundColor: colors.card, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10, fontSize: 15 },
  chips: { paddingHorizontal: 12, paddingVertical: 6 },
  count: { color: colors.muted, fontSize: 12, paddingHorizontal: 14, paddingVertical: 6 },
  error: { color: '#b91c1c', paddingHorizontal: 14, paddingBottom: 8 },
  empty: { textAlign: 'center', color: colors.muted, padding: 32 },
});
