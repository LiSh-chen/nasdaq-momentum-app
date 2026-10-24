import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, ScrollView, Share, StyleSheet, Text, TextInput, View } from 'react-native';
import { LEVEL_NAMES, SOURCE_NAMES } from '../api/mitLearn';
import { PROVIDERS, searchCatalog } from '../api/search';
import BrowserRunner from '../components/BrowserRunner';
import { Chip, colors } from '../ui';
import CourseCard from './CourseCard';


// 已選的項目即使不在這次的篩選統計裡，也要留在列上讓使用者能取消
// 臺大、陽明交大沒有關鍵字時，提供幾個常見的搜尋起點
const SUGGESTIONS = ['微積分', '物理', '化學', '統計', '程式', '英文'];

const withSelected = (list = [], selected) =>
  selected && !list.some((f) => f.key === selected) ? [{ key: selected, count: 0 }, ...list] : list;

export default function Explore({ onOpenCourse }) {
  const [query, setQuery] = useState('');
  const [provider, setProvider] = useState('all');
  const [filters, setFilters] = useState({ offeredBy: null, level: null, topic: null, hasVideos: false, sort: null });
  const [state, setState] = useState({ items: [], count: 0, facets: {}, loading: true, error: null, next: null, failed: 0, browse: false, defaultQuery: null, meta: null });
  const reqId = useRef(0);
  const browser = useRef(null);

  const run = useCallback(
    async (cursor, append) => {
      const id = ++reqId.current;
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        // 第一頁不帶頁碼，由各來源決定起點（MIT 是 offset 0，臺大與陽明交大是第 1 頁）
        const r = await searchCatalog({ provider, q: query, ...filters, cursor: append ? cursor : undefined, browser: browser.current });
        if (id !== reqId.current) return; // 已有較新的搜尋，丟棄舊結果
        setState((s) => ({
          items: append ? [...s.items, ...r.items] : r.items,
          count: r.count,
          facets: append ? s.facets : r.facets ?? {},
          next: r.next ?? null,
          failed: r.failed ?? 0,
          browse: Boolean(r.browse),
          defaultQuery: r.defaultQuery ?? null,
          meta: r.meta ?? null,
          loading: false,
          error: null,
        }));
      } catch (e) {
        if (id !== reqId.current || /superseded/.test(e?.message)) return;
        setState((s) => ({ ...s, loading: false, meta: null, error: provider === 'ntu' ? '臺大網站沒有回應或驗證未完成，請稍後再試' : '載入失敗，請檢查網路後再試一次' }));
      }
    },
    [provider, query, filters]
  );

  useEffect(() => {
    const t = setTimeout(() => run(0, false), query ? 400 : 0);
    return () => clearTimeout(t);
  }, [run]);

  const pickProvider = (id) => {
    setProvider(id);
    setFilters({ offeredBy: null, level: null, topic: null, hasVideos: false, sort: null });
  };
  const set = (key, value) => setFilters((f) => ({ ...f, [key]: f[key] === value ? null : value }));
  const { facets } = state;
  // 臺大、陽明交大是解析網頁取得的結果：有關鍵字卻沒有結果、或出錯時，提供回報（附查詢資訊，方便找出原因）
  const showReport = !state.loading && (provider === 'nycu' || provider === 'ntu') && (Boolean(state.error) || (state.items.length === 0 && query.trim().length > 0));
  const report = () => Share.share({ message: JSON.stringify({ provider, query, error: state.error, meta: state.meta }, null, 1) });

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
        {PROVIDERS.map((p) => (
          <Chip key={p.id} label={p.label} active={provider === p.id} onPress={() => pickProvider(p.id)} />
        ))}
      </ScrollView>
      {provider === 'mit' ? (
        <View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        <Chip label="所有 MIT 課程" active={!filters.offeredBy} onPress={() => setFilters((f) => ({ ...f, offeredBy: null }))} />
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
        </View>
      ) : null}
      <Text style={styles.count}>{state.loading && !state.items.length ? '搜尋中…' : `共 ${state.count} 門課程`}</Text>
      {state.error ? <Text style={styles.error}>{state.error}</Text> : null}
      {state.failed ? <Text style={styles.note}>有 {state.failed} 個來源暫時無法連線，已先顯示其他來源的結果。</Text> : null}
      {provider === 'nycu' && !query.trim() && state.defaultQuery ? <Text style={styles.note}>目前顯示的是「{state.defaultQuery}」相關課程，輸入關鍵字可搜尋其他課程。</Text> : null}
      {provider === 'ntu' && !query.trim() && state.browse ? <Text style={styles.note}>目前顯示的是最新與熱門課程，輸入關鍵字可搜尋全部課程。</Text> : null}
      {provider === 'ntu' ? <Text style={styles.note}>臺大網站需要驗證，搜尋會在背景以內建瀏覽器進行，可能需要幾秒鐘。</Text> : null}
      {(provider === 'nycu' || provider === 'ntu') && !query.trim() ? (
        <View style={styles.suggest}>
          <Text style={styles.note}>試試看：</Text>
          {SUGGESTIONS.map((w) => (
            <Pressable key={w} onPress={() => setQuery(w)} style={styles.suggestChip}><Text style={styles.suggestText}>{w}</Text></Pressable>
          ))}
        </View>
      ) : null}
      <Text style={styles.note}>Coursera、edX 等網站不開放自動搜尋，請到「網站」分頁瀏覽後按「整理」。</Text>
    </View>
  );

  return (
    <View style={{ flex: 1 }}>
    <FlatList
      style={styles.list}
      data={state.items}
      keyExtractor={(c) => String(c.id)}
      renderItem={({ item }) => <CourseCard course={item} onPress={onOpenCourse} />}
      ListHeaderComponent={header}
      ListFooterComponent={
        state.loading ? <ActivityIndicator style={{ margin: 16 }} /> : showReport ? (
          <Pressable onPress={report} style={styles.reportBtn}><Text style={styles.reportText}>回報這個問題</Text></Pressable>
        ) : null
      }
      ListEmptyComponent={!state.loading && !state.error ? <Text style={styles.empty}>{query.trim() ? '找不到符合的課程' : '目前沒有可顯示的課程'}</Text> : null}
      onEndReached={() => state.next !== null && !state.loading && run(state.next, true)}
      onEndReachedThreshold={0.6}
      keyboardShouldPersistTaps="handled"
    />
    <BrowserRunner ref={browser} />
    </View>
  );
}

const styles = StyleSheet.create({
  list: { flex: 1, backgroundColor: colors.bg },
  searchBox: { padding: 12, paddingBottom: 4 },
  input: { backgroundColor: colors.card, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10, fontSize: 15 },
  chips: { paddingHorizontal: 12, paddingVertical: 6 },
  suggest: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', paddingHorizontal: 14, paddingBottom: 8, gap: 6 },
  suggestChip: { backgroundColor: colors.primarySoft, borderRadius: 14, paddingHorizontal: 12, paddingVertical: 5 },
  suggestText: { color: colors.primary, fontSize: 13 },
  reportBtn: { alignSelf: 'center', padding: 14 },
  reportText: { color: colors.primary, fontSize: 13 },
  count: { color: colors.muted, fontSize: 12, paddingHorizontal: 14, paddingVertical: 6 },
  note: { color: colors.muted, fontSize: 11, paddingHorizontal: 14, paddingBottom: 6 },
  error: { color: '#b91c1c', paddingHorizontal: 14, paddingBottom: 8 },
  empty: { textAlign: 'center', color: colors.muted, padding: 32 },
});
