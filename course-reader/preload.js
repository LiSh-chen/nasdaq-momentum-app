const { contextBridge, ipcRenderer, } = require('electron');

contextBridge.exposeInMainWorld('api', {
  listSites: () => ipcRenderer.invoke('sites:list'),
  addSite: (site) => ipcRenderer.invoke('sites:add', site),
  removeSite: (id) => ipcRenderer.invoke('sites:remove', id),
  openSite: (id) => ipcRenderer.invoke('site:open', id),
  logout: (id) => ipcRenderer.invoke('site:logout', id),
  nav: (action) => ipcRenderer.invoke('nav', action),
  onNavState: (cb) => ipcRenderer.on('nav-state', (_e, state) => cb(state)),
});
