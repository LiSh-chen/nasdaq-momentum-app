import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert, BackHandler, FlatList, Modal, Pressable, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { WebView } from 'react-native-webview';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CookieManager from '@react-native-cookies/cookies';
import defaultSites from './sites.default.json';

const SITES_KEY = 'sites';
// 沒有到期日的 session cookie 在 App 被關閉後會消失；延長後登入才能跨次啟動保留
const COOKIE_LIFETIME_MS = 30 * 24 * 60 * 60 * 1000;

async function persistSessionCookies(url) {
  try {
    const cookies = await CookieManager.get(url);
    const expires = new Date(Date.now() + COOKIE_LIFETIME_MS).toISOString();
    await Promise.all(
      Object.values(cookies)
        .filter((c) => !c.expires)
        .map((c) => CookieManager.set(url, { ...c, expires }))
    );
    await CookieManager.flush();
  } catch {}
}

async function clearCookiesFor(url) {
  try {
    const cookies = await CookieManager.get(url);
    const past = new Date(0).toISOString();
    await Promise.all(
      Object.values(cookies).map((c) => CookieManager.set(url, { ...c, expires: past }))
    );
    await CookieManager.flush();
  } catch {}
}

function Main() {
  const [sites, setSites] = useState(defaultSites);
  const [active, setActive] = useState(null);
  const [menuOpen, setMenuOpen] = useState(true);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState({ name: '', url: 'https://' });
  const [nav, setNav] = useState({ canGoBack: false, canGoForward: false, url: '' });
  const [reloadKey, setReloadKey] = useState(0);
  const web = useRef(null);

  useEffect(() => {
    AsyncStorage.getItem(SITES_KEY).then((raw) => raw && setSites(JSON.parse(raw)));
  }, []);

  const saveSites = (next) => {
    setSites(next);
    AsyncStorage.setItem(SITES_KEY, JSON.stringify(next));
  };

  // Android 實體返回鍵：先回上一頁，沒有上一頁再回到網站清單
  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (adding) { setAdding(false); return true; }
      if (active && !menuOpen) {
        if (nav.canGoBack) web.current?.goBack();
        else setMenuOpen(true);
        return true;
      }
      return false;
    });
    return () => sub.remove();
  }, [active, menuOpen, adding, nav.canGoBack]);

  const open = (site) => {
    setActive(site);
    setMenuOpen(false);
    setNav({ canGoBack: false, canGoForward: false, url: site.url });
  };

  const addSite = () => {
    try { new URL(draft.url); } catch { return Alert.alert('網址格式不正確'); }
    if (!draft.name.trim()) return Alert.alert('請輸入網站名稱');
    saveSites([...sites, { id: `custom-${Date.now()}`, name: draft.name.trim(), url: draft.url }]);
    setDraft({ name: '', url: 'https://' });
    setAdding(false);
  };

  const removeSite = (site) =>
    Alert.alert(`移除「${site.name}」？`, '同時會清除此網站的登入資料。', [
      { text: '取消', style: 'cancel' },
      {
        text: '移除', style: 'destructive',
        onPress: async () => {
          await clearCookiesFor(site.url);
          saveSites(sites.filter((s) => s.id !== site.id));
          if (active?.id === site.id) setActive(null);
        },
      },
    ]);

  const logout = () =>
    Alert.alert('登出此網站？', '會清除此網站的登入資料。', [
      { text: '取消', style: 'cancel' },
      {
        text: '登出', style: 'destructive',
        onPress: async () => {
          await clearCookiesFor(active.url);
          setReloadKey((k) => k + 1);
        },
      },
    ]);

  const onNavChange = useCallback((s) => {
    setNav({ canGoBack: s.canGoBack, canGoForward: s.canGoForward, url: s.url });
    if (!s.loading) persistSessionCookies(s.url);
  }, []);

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />
      <View style={styles.toolbar}>
        <Btn label="☰" onPress={() => setMenuOpen(true)} />
        <Btn label="←" disabled={!nav.canGoBack} onPress={() => web.current?.goBack()} />
        <Btn label="→" disabled={!nav.canGoForward} onPress={() => web.current?.goForward()} />
        <Btn label="⟳" disabled={!active} onPress={() => web.current?.reload()} />
        <Text style={styles.title} numberOfLines={1}>{active?.name ?? '課程閱讀器'}</Text>
        <Btn label="登出" disabled={!active} onPress={logout} />
      </View>

      {active ? (
        <WebView
          key={`${active.id}-${reloadKey}`}
          ref={web}
          source={{ uri: active.url }}
          sharedCookiesEnabled
          thirdPartyCookiesEnabled
          domStorageEnabled
          javaScriptEnabled
          allowsInlineMediaPlayback
          mediaPlaybackRequiresUserAction={false}
          allowsFullscreenVideo
          setSupportMultipleWindows={false}
          onNavigationStateChange={onNavChange}
          style={styles.web}
        />
      ) : (
        <View style={styles.empty}><Text style={styles.muted}>點左上角 ☰ 選擇課程網站</Text></View>
      )}

      <Modal visible={menuOpen} animationType="slide" onRequestClose={() => active && setMenuOpen(false)}>
        <SafeAreaView style={styles.menu}>
          <View style={styles.menuHead}>
            <Text style={styles.menuTitle}>📚 課程網站</Text>
            {active && <Btn label="關閉" onPress={() => setMenuOpen(false)} />}
          </View>
          <FlatList
            data={sites}
            keyExtractor={(s) => s.id}
            renderItem={({ item }) => (
              <Pressable
                style={[styles.row, item.id === active?.id && styles.rowActive]}
                onPress={() => open(item)}
                onLongPress={() => removeSite(item)}
              >
                <Text style={styles.rowName}>{item.name}</Text>
                <Text style={styles.rowUrl} numberOfLines={1}>{item.url}</Text>
              </Pressable>
            )}
            ListFooterComponent={<Text style={styles.hint}>長按網站可移除並清除登入資料</Text>}
          />
          <Pressable style={styles.addBtn} onPress={() => setAdding(true)}>
            <Text style={styles.addBtnText}>＋ 新增網站</Text>
          </Pressable>
        </SafeAreaView>

        <Modal visible={adding} transparent animationType="fade" onRequestClose={() => setAdding(false)}>
          <View style={styles.backdrop}>
            <View style={styles.dialog}>
              <TextInput style={styles.input} placeholder="網站名稱" value={draft.name}
                onChangeText={(name) => setDraft({ ...draft, name })} />
              <TextInput style={styles.input} placeholder="https://..." value={draft.url}
                autoCapitalize="none" keyboardType="url"
                onChangeText={(url) => setDraft({ ...draft, url })} />
              <View style={styles.dialogRow}>
                <Btn label="取消" dark onPress={() => setAdding(false)} />
                <Btn label="新增" dark onPress={addSite} />
              </View>
            </View>
          </View>
        </Modal>
      </Modal>
    </SafeAreaView>
  );
}

function Btn({ label, onPress, disabled, dark }) {
  return (
    <Pressable onPress={onPress} disabled={disabled} style={[styles.btn, disabled && { opacity: 0.35 }]}>
      <Text style={[styles.btnText, dark && { color: '#2563eb' }]}>{label}</Text>
    </Pressable>
  );
}

export default function App() {
  return <SafeAreaProvider><Main /></SafeAreaProvider>;
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#111827' },
  toolbar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 6, height: 48 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
  btn: { paddingHorizontal: 10, paddingVertical: 8 },
  btnText: { color: '#fff', fontSize: 16 },
  web: { flex: 1 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#f3f4f6' },
  muted: { color: '#6b7280' },
  menu: { flex: 1, backgroundColor: '#fff' },
  menuHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16 },
  menuTitle: { fontSize: 20, fontWeight: '700' },
  row: { paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#d1d5db' },
  rowActive: { backgroundColor: '#dbeafe' },
  rowName: { fontSize: 16, fontWeight: '600' },
  rowUrl: { color: '#6b7280', marginTop: 2 },
  hint: { color: '#9ca3af', textAlign: 'center', padding: 16, fontSize: 12 },
  addBtn: { margin: 16, padding: 14, borderRadius: 8, backgroundColor: '#2563eb', alignItems: 'center' },
  addBtnText: { color: '#fff', fontWeight: '600' },
  backdrop: { flex: 1, backgroundColor: '#0008', justifyContent: 'center', padding: 24 },
  dialog: { backgroundColor: '#fff', borderRadius: 12, padding: 16, gap: 10 },
  input: { borderWidth: 1, borderColor: '#d1d5db', borderRadius: 6, padding: 10 },
  dialogRow: { flexDirection: 'row', justifyContent: 'flex-end' },
});
