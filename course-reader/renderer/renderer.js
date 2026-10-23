const listEl = document.getElementById('list');
const emptyEl = document.getElementById('empty');
let sites = [];
let activeId = null;

function render() {
  listEl.replaceChildren(
    ...sites.map((site) => {
      const row = document.createElement('div');
      row.className = 'site' + (site.id === activeId ? ' active' : '');
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = site.name;
      const del = document.createElement('button');
      del.textContent = '✕';
      del.title = '移除並清除登入資料';
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`移除「${site.name}」並清除其登入資料？`)) return;
        sites = await window.api.removeSite(site.id);
        if (activeId === site.id) {
          activeId = null;
          emptyEl.style.display = 'grid';
        }
        render();
      };
      row.append(name, del);
      row.onclick = () => open(site.id);
      return row;
    })
  );
}

async function open(id) {
  activeId = id;
  emptyEl.style.display = 'none';
  render();
  await window.api.openSite(id);
}

document.getElementById('add').onsubmit = async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  sites = await window.api.addSite({ name: form.get('name'), url: form.get('url') });
  e.target.reset();
  render();
};

document.getElementById('back').onclick = () => window.api.nav('back');
document.getElementById('forward').onclick = () => window.api.nav('forward');
document.getElementById('reload').onclick = () => window.api.nav('reload');
document.getElementById('logout').onclick = () => {
  if (activeId && confirm('清除此網站的登入資料？')) window.api.logout(activeId);
};

window.api.onNavState(({ url, canGoBack, canGoForward }) => {
  document.getElementById('url').textContent = url;
  document.getElementById('back').disabled = !canGoBack;
  document.getElementById('forward').disabled = !canGoForward;
});

window.api.listSites().then((s) => {
  sites = s;
  render();
});
