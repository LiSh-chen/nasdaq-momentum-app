const { app, BrowserWindow, WebContentsView, ipcMain, session } = require('electron');
const fs = require('fs');
const path = require('path');

const SIDEBAR_W = 240;
const TOOLBAR_H = 44;
// 沒有到期日的 session cookie 預設關閉 App 就消失；延長後登入才能跨次啟動保留
const COOKIE_LIFETIME_S = 60 * 60 * 24 * 30;

let win;
const views = new Map(); // siteId -> WebContentsView
let activeId = null;

const sitesFile = () => path.join(app.getPath('userData'), 'sites.json');

function loadSites() {
  try {
    return JSON.parse(fs.readFileSync(sitesFile(), 'utf8'));
  } catch {
    return JSON.parse(fs.readFileSync(path.join(__dirname, 'sites.default.json'), 'utf8'));
  }
}

function saveSites(sites) {
  fs.writeFileSync(sitesFile(), JSON.stringify(sites, null, 2));
}

const partitionOf = (id) => `persist:site-${id}`;

function keepCookiesAlive(ses) {
  ses.cookies.on('changed', (_e, cookie, _cause, removed) => {
    if (removed || cookie.expirationDate) return;
    const scheme = cookie.secure ? 'https' : 'http';
    const host = cookie.domain.replace(/^\./, '');
    ses.cookies
      .set({
        url: `${scheme}://${host}${cookie.path}`,
        name: cookie.name,
        value: cookie.value,
        domain: cookie.domain,
        path: cookie.path,
        secure: cookie.secure,
        httpOnly: cookie.httpOnly,
        sameSite: cookie.sameSite,
        expirationDate: Math.floor(Date.now() / 1000) + COOKIE_LIFETIME_S,
      })
      .catch(() => {});
  });
}

function layout() {
  if (!win) return;
  const { width, height } = win.getContentBounds();
  const view = views.get(activeId);
  if (view) {
    view.setBounds({
      x: SIDEBAR_W,
      y: TOOLBAR_H,
      width: Math.max(0, width - SIDEBAR_W),
      height: Math.max(0, height - TOOLBAR_H),
    });
  }
}

function sendNav() {
  const view = views.get(activeId);
  if (!view || !win) return;
  const wc = view.webContents;
  win.webContents.send('nav-state', {
    id: activeId,
    url: wc.getURL(),
    canGoBack: wc.navigationHistory.canGoBack(),
    canGoForward: wc.navigationHistory.canGoForward(),
  });
}

function createView(site) {
  const ses = session.fromPartition(partitionOf(site.id));
  keepCookiesAlive(ses);
  const view = new WebContentsView({
    webPreferences: { session: ses, sandbox: true, contextIsolation: true },
  });
  const wc = view.webContents;
  // 站內開新視窗的連結一律在同一個分頁開，不跳出 App
  wc.setWindowOpenHandler(({ url }) => {
    wc.loadURL(url);
    return { action: 'deny' };
  });
  wc.on('did-navigate', sendNav);
  wc.on('did-navigate-in-page', sendNav);
  wc.on('page-title-updated', sendNav);
  wc.loadURL(site.url);
  views.set(site.id, view);
  return view;
}

function activate(id) {
  const site = loadSites().find((s) => s.id === id);
  if (!site) return;
  const current = views.get(activeId);
  if (current) win.contentView.removeChildView(current);
  const view = views.get(id) || createView(site);
  win.contentView.addChildView(view);
  activeId = id;
  layout();
  sendNav();
}

function destroyView(id) {
  const view = views.get(id);
  if (!view) return;
  if (id === activeId) {
    win.contentView.removeChildView(view);
    activeId = null;
  }
  view.webContents.close();
  views.delete(id);
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 820,
    title: '課程閱讀器',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.on('resize', layout);
}

ipcMain.handle('sites:list', () => loadSites());

ipcMain.handle('sites:add', (_e, { name, url }) => {
  const sites = loadSites();
  const id = `custom-${Date.now()}`;
  sites.push({ id, name, url });
  saveSites(sites);
  return sites;
});

ipcMain.handle('sites:remove', async (_e, id) => {
  destroyView(id);
  await session.fromPartition(partitionOf(id)).clearStorageData();
  const sites = loadSites().filter((s) => s.id !== id);
  saveSites(sites);
  return sites;
});

ipcMain.handle('site:open', (_e, id) => activate(id));

// 登出：清掉該站的 cookie 與儲存資料，其他站不受影響
ipcMain.handle('site:logout', async (_e, id) => {
  await session.fromPartition(partitionOf(id)).clearStorageData();
  views.get(id)?.webContents.reload();
});

ipcMain.handle('nav', (_e, action) => {
  const wc = views.get(activeId)?.webContents;
  if (!wc) return;
  if (action === 'back') wc.navigationHistory.goBack();
  else if (action === 'forward') wc.navigationHistory.goForward();
  else if (action === 'reload') wc.reload();
});

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
