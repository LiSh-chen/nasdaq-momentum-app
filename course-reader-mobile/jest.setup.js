jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'));
jest.mock('@react-native-cookies/cookies', () => ({
  get: jest.fn(), set: jest.fn(() => Promise.resolve(true)), flush: jest.fn(() => Promise.resolve()),
}));
jest.mock('react-native-webview', () => {
  const React = require('react');
  const { View } = require('react-native');
  const WebView = React.forwardRef((props, ref) => {
    React.useImperativeHandle(ref, () => ({ goBack: jest.fn(), goForward: jest.fn(), reload: jest.fn(), injectJavaScript: jest.fn() }));
    return <View testID="webview" {...props} />;
  });
  return { WebView };
});
jest.mock('react-native-safe-area-context', () =>
  require('react-native-safe-area-context/jest/mock').default);
