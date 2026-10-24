// 從 Android 版（course-reader-mobile）同步程式碼：兩邊共用同一份 src，這裡只放 iOS 專屬的設定檔。
import { cpSync, rmSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = join(dirname(fileURLToPath(import.meta.url)), '..');
const from = join(here, '..', 'course-reader-mobile');

rmSync(join(here, 'src'), { recursive: true, force: true });
cpSync(join(from, 'src'), join(here, 'src'), {
  recursive: true,
  // WITH_TESTS=1 時一併帶入單元測試（用來在新版 React 下驗證同一份程式）
  filter: (p) => !/\.live\.js$/.test(p) && !p.endsWith('/src/live') && (process.env.WITH_TESTS === '1' || !/\.test\.js$/.test(p)),
});
if (process.env.WITH_TESTS === '1') {
  const setup = readFileSync(join(from, 'jest.setup.js'), 'utf8')
    // iOS 版沒有原生 cookies 套件（以 stub 取代），測試時改用 virtual mock
    .replace(/(jest\.mock\('@react-native-cookies\/cookies', \(\) => \(\{[\s\S]*?\}\))\);/, '$1, { virtual: true });');
  writeFileSync(join(here, 'jest.setup.js'), setup);
}
cpSync(join(from, 'App.js'), join(here, 'App.js'));
cpSync(join(from, 'sites.default.json'), join(here, 'sites.default.json'));
console.log('synced from', from, existsSync(join(here, 'src')) ? '(ok)' : '(missing src)');
