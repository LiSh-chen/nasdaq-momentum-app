import React, { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { WebView } from 'react-native-webview';
import { EXTRACT_JS } from '../content/extractor';
import { colors } from '../ui';

const MAX_CHALLENGE_TRIES = 20; // 約 30 秒
const MAX_NOT_READY_TRIES = 6;

// 隱藏的 App 內建瀏覽器：載入網頁、擷取內容後以 Promise 回傳。
// 與一般瀏覽相同，網站的驗證會自動通過；需要使用者操作時才把網頁顯示出來。
const BrowserRunner = forwardRef(function BrowserRunner(_props, ref) {
  const web = useRef(null);
  const job = useRef(null);
  const [state, setState] = useState({ url: null, key: 0, challenge: false });

  const settle = (kind, value) => {
    const j = job.current;
    if (!j) return;
    job.current = null;
    clearTimeout(j.timer);
    setState({ url: null, key: j.key, challenge: false });
    j[kind](value);
  };

  useImperativeHandle(ref, () => ({
    run(url, { ready, timeoutMs = 45000 } = {}) {
      return new Promise((resolve, reject) => {
        if (job.current) {
          clearTimeout(job.current.timer);
          job.current.reject(new Error('superseded'));
        }
        const key = Date.now() + Math.random();
        const timer = setTimeout(() => settle('reject', new Error('timeout')), timeoutMs);
        job.current = { resolve, reject, timer, key, ready, challengeTries: 0, notReadyTries: 0 };
        setState({ url, key, challenge: false });
      });
    },
  }), []);

  const extract = () => web.current?.injectJavaScript(EXTRACT_JS);

  const onMessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.nativeEvent.data); } catch { return; }
    const j = job.current;
    if (!j) return;
    if (msg.type === 'content') {
      // 頁面內容是載入後才產生的：還沒出現預期的內容時多等幾次
      if (j.ready && !j.ready(msg) && j.notReadyTries < MAX_NOT_READY_TRIES) {
        j.notReadyTries += 1;
        setTimeout(extract, 1500);
        return;
      }
      settle('resolve', msg);
    } else if (msg.type === 'challenge') {
      setState((s) => ({ ...s, challenge: true }));
      if (++j.challengeTries < MAX_CHALLENGE_TRIES) setTimeout(extract, 1500);
    } else if (msg.type === 'error') {
      settle('reject', new Error(msg.message));
    }
  };

  if (!state.url) return null;
  return (
    <View style={state.challenge ? styles.overlay : styles.hidden} pointerEvents={state.challenge ? 'auto' : 'none'} testID="runner">
      {state.challenge ? (
        <View style={styles.bar}>
          <Text style={styles.barText}>網站正在進行驗證，通過後會自動繼續；若出現勾選框請完成驗證。</Text>
          <Pressable onPress={() => settle('reject', new Error('cancelled'))}><Text style={styles.cancel}>取消</Text></Pressable>
        </View>
      ) : null}
      <WebView
        key={state.key}
        ref={web}
        source={{ uri: state.url }}
        sharedCookiesEnabled
        thirdPartyCookiesEnabled
        domStorageEnabled
        javaScriptEnabled
        setSupportMultipleWindows={false}
        onLoadEnd={() => setTimeout(extract, 800)}
        onMessage={onMessage}
        style={{ flex: 1 }}
      />
    </View>
  );
});

export default BrowserRunner;

const styles = StyleSheet.create({
  hidden: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, opacity: 0 },
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: '#fff', zIndex: 10 },
  bar: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10, backgroundColor: colors.dark },
  barText: { flex: 1, color: '#e5e7eb', fontSize: 12 },
  cancel: { color: '#93c5fd', fontSize: 14 },
});
