// Expo Go 沒有 @react-native-cookies/cookies。iOS 的 WKWebView 本身會保存 cookie，
// 這個空實作只讓「延長 session cookie」與「單站登出」這兩個輔助功能安靜地略過。
const noop = async () => ({});
export default { get: noop, set: async () => true, flush: async () => {}, clearAll: async () => true };
