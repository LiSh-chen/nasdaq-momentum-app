import React, { useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { persistSessionCookies } from '../cookies';
import { Btn, colors } from '../ui';

// 需登入的來源（如 MITx）：在 App 內開啟網頁，並記住登入狀態
export default function WebPage({ title, url, onClose, onHome }) {
  const web = useRef(null);
  const [nav, setNav] = useState({ canGoBack: false });
  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.bar}>
        <Btn label="← 返回" onPress={onClose} />
        {onHome ? <Btn label="首頁" onPress={onHome} /> : null}
        <Btn label="上一頁" disabled={!nav.canGoBack} onPress={() => web.current?.goBack()} />
        <Text style={styles.title} numberOfLines={1}>{title}</Text>
        <Btn label="⟳" onPress={() => web.current?.reload()} />
      </View>
      <WebView
        ref={web}
        source={{ uri: url }}
        sharedCookiesEnabled
        thirdPartyCookiesEnabled
        domStorageEnabled
        javaScriptEnabled
        allowsInlineMediaPlayback
        mediaPlaybackRequiresUserAction={false}
        allowsFullscreenVideo
        setSupportMultipleWindows={false}
        onNavigationStateChange={(s) => {
          setNav({ canGoBack: s.canGoBack });
          if (!s.loading) persistSessionCookies(s.url);
        }}
        style={{ flex: 1 }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.dark },
  bar: { flexDirection: 'row', alignItems: 'center', height: 48, paddingHorizontal: 6 },
  title: { flex: 1, color: '#e5e7eb', fontWeight: '600', textAlign: 'center', marginHorizontal: 4 },
});
