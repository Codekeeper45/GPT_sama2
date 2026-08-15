(() => {
  'use strict';
  const TOTAL = 140;
  const STORAGE_KEY = 'miside-manga-reader-v2';
  const FIRST_VISIT_KEY = 'miside-manga-welcomed-v1';
  let currentPage = 1;
  let mode = 'single';
  let zoom = 'width';
  let renderToken = 0;
  let scrollObserver = null;
  let saveTimer = null;

  const $ = (id) => document.getElementById(id);
  const els = {};
  ['pageLabel','percentLabel','progressFill','pageImage','pageStage','loading','prevBtn','nextBtn','pagePill','drawer','drawerBackdrop','menuBtn','shareBtn','closeDrawer','pageBtn','pageGrid','savedPageText','continueBtn','continueCard','singleModeBtn','scrollModeBtn','widthModeBtn','fitModeBtn','resetBtn','singleMode','scrollMode','welcome','startBtn','welcomeContinueBtn','toast','topbar','bottomNav'].forEach(id => els[id] = $(id));

  const clamp = (n,min,max) => Math.max(min,Math.min(max,n));
  const state = () => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; } };
  function saveState(extra={}) {
    const old = state();
    const next = {...old, page:currentPage, mode, zoom, updatedAt:Date.now(), ...extra};
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    updateDrawerProgress();
  }
  function parseHash() {
    const m = location.hash.match(/(?:page|p)=(\d+)/i) || location.hash.match(/^#(\d+)$/);
    return m ? clamp(Number(m[1]),1,TOTAL) : null;
  }
  function updateHash(page) { history.replaceState(null,'',`#page=${page}`); }
  function toast(msg) { els.toast.textContent=msg; els.toast.classList.add('show'); clearTimeout(toast.t); toast.t=setTimeout(()=>els.toast.classList.remove('show'),1800); }
  const assetMeta = document.querySelector('meta[name="manga-asset-base"]');
  const ASSET_BASE = (assetMeta?.content || '.').replace(/\/$/,'');
  function pageUrl(page) { return Promise.resolve(`${ASSET_BASE}/pages/page-${String(page).padStart(3,'0')}.webp`); }
  function prefetch(page) {
    [page-1,page+1,page+2].filter(p=>p>=1&&p<=TOTAL).forEach(p=>{ const im=new Image(); im.src=`${ASSET_BASE}/pages/page-${String(p).padStart(3,'0')}.webp`; });
  }
  function updateChrome() {
    const percent=Math.round(currentPage/TOTAL*100);
    els.pageLabel.textContent=`${currentPage} / ${TOTAL}`; els.pagePill.textContent=currentPage; els.percentLabel.textContent=`${percent}%`; els.progressFill.style.width=`${currentPage/TOTAL*100}%`;
    els.prevBtn.disabled=currentPage<=1; els.nextBtn.disabled=currentPage>=TOTAL;
    document.title=`Страница ${currentPage} — Одиночество через стекло`;
    document.querySelectorAll('.page-grid button').forEach(b=>{ const n=Number(b.dataset.page); b.classList.toggle('current',n===currentPage); b.classList.toggle('read',n<=Math.max(currentPage,state().page||1)); });
  }
  async function showPage(page,{scrollTop=true}={}) {
    page=clamp(Number(page)||1,1,TOTAL); currentPage=page; updateChrome(); updateHash(page); saveState();
    if(mode==='scroll'){ scrollToScrollPage(page); return; }
    const token=++renderToken; els.loading.classList.remove('hidden'); els.pageImage.style.opacity='.12';
    try { const url=await pageUrl(page); if(token!==renderToken)return; els.pageImage.onload=()=>{ els.loading.classList.add('hidden'); els.pageImage.style.opacity='1'; }; els.pageImage.src=url; els.pageImage.alt=`Страница ${page} из ${TOTAL}`; if(els.pageImage.complete){els.loading.classList.add('hidden');els.pageImage.style.opacity='1'} }
    catch(e){ els.loading.classList.add('hidden'); toast('Не удалось загрузить страницу'); console.error(e); }
    if(scrollTop) window.scrollTo({top:0,behavior:'instant'}); prefetch(page);
  }
  function go(delta){ if(mode==='scroll') { showPage(currentPage+delta); return; } showPage(currentPage+delta); }
  function setMode(next){ if(next===mode)return; mode=next; els.singleModeBtn.classList.toggle('active',mode==='single'); els.scrollModeBtn.classList.toggle('active',mode==='scroll'); els.singleMode.classList.toggle('hidden',mode!=='single'); els.scrollMode.classList.toggle('hidden',mode!=='scroll'); saveState(); closeDrawer(); if(mode==='scroll'){ buildScroll(); requestAnimationFrame(()=>scrollToScrollPage(currentPage,false)); } else { if(scrollObserver){scrollObserver.disconnect();scrollObserver=null} showPage(currentPage); } }
  function setZoom(next){ zoom=next; els.widthModeBtn.classList.toggle('active',zoom==='width'); els.fitModeBtn.classList.toggle('active',zoom==='fit'); els.pageStage.classList.toggle('fit',zoom==='fit'); saveState(); }
  function openDrawer(){ els.drawer.classList.add('open'); els.drawer.setAttribute('aria-hidden','false'); document.body.style.overflow='hidden'; updateDrawerProgress(); }
  function closeDrawer(){ els.drawer.classList.remove('open'); els.drawer.setAttribute('aria-hidden','true'); document.body.style.overflow=''; }
  function updateDrawerProgress(){ const s=state(); const p=clamp(s.page||1,1,TOTAL); els.savedPageText.textContent=`${p} / ${TOTAL}`; els.continueCard.classList.toggle('hidden',p<=1); els.welcomeContinueBtn.classList.toggle('hidden',p<=1); if(p>1) els.welcomeContinueBtn.textContent=`Продолжить с ${p}`; }
  function buildGrid(){ const frag=document.createDocumentFragment(); for(let i=1;i<=TOTAL;i++){ const b=document.createElement('button'); b.textContent=i; b.dataset.page=i; b.addEventListener('click',()=>{closeDrawer();showPage(i)}); frag.appendChild(b); } els.pageGrid.appendChild(frag); updateChrome(); }
  async function share(){ const url=`${location.href.split('#')[0]}#page=${currentPage}`; const data={title:'Одиночество через стекло',text:`Манга — страница ${currentPage} из ${TOTAL}`,url}; try{ if(navigator.share) await navigator.share(data); else { await navigator.clipboard.writeText(url); toast('Ссылка скопирована'); } }catch(e){ if(e?.name!=='AbortError') toast('Не удалось поделиться'); } }
  function reset(){ if(!confirm('Сбросить сохранённый прогресс чтения?'))return; localStorage.removeItem(STORAGE_KEY); currentPage=1; mode='single'; zoom='width'; setZoom('width'); setMode('single'); showPage(1); toast('Прогресс сброшен'); }

  function buildScroll(){
    els.scrollMode.innerHTML=''; const frag=document.createDocumentFragment();
    for(let i=1;i<=TOTAL;i++){ const d=document.createElement('article'); d.className='scroll-page'; d.dataset.page=i; d.id=`scroll-page-${i}`; const ph=document.createElement('div'); ph.className='scroll-placeholder'; ph.textContent=`${i}`; d.appendChild(ph); const badge=document.createElement('span'); badge.className='badge'; badge.textContent=`${i} / ${TOTAL}`; d.appendChild(badge); frag.appendChild(d); }
    els.scrollMode.appendChild(frag);
    scrollObserver=new IntersectionObserver(entries=>{
      let most=null; entries.forEach(e=>{ if(e.isIntersecting){ loadScrollImage(e.target); if(!most || e.intersectionRatio>most.intersectionRatio)most=e; } });
      if(most){ const p=Number(most.target.dataset.page); if(p!==currentPage){ currentPage=p; updateChrome(); updateHash(p); clearTimeout(saveTimer); saveTimer=setTimeout(saveState,200); } }
    },{root:null,rootMargin:'120% 0px 120% 0px',threshold:[.01,.25,.55,.8]});
    document.querySelectorAll('.scroll-page').forEach(el=>scrollObserver.observe(el));
  }
  async function loadScrollImage(el){ if(el.dataset.loaded)return; el.dataset.loaded='1'; const p=Number(el.dataset.page); try{ const url=await pageUrl(p); const img=new Image(); img.alt=`Страница ${p}`; img.loading='lazy'; img.draggable=false; img.src=url; img.onload=()=>{ const ph=el.querySelector('.scroll-placeholder'); if(ph)ph.remove(); el.insertBefore(img,el.firstChild); }; }catch(e){el.dataset.loaded='';} }
  function scrollToScrollPage(page,smooth=true){ const el=$(`scroll-page-${page}`); if(!el)return; el.scrollIntoView({block:'start',behavior:smooth?'smooth':'instant'}); loadScrollImage(el); }

  function bind(){
    els.prevBtn.onclick=()=>go(-1); els.nextBtn.onclick=()=>go(1); els.menuBtn.onclick=openDrawer; els.pageBtn.onclick=openDrawer; els.closeDrawer.onclick=closeDrawer; els.drawerBackdrop.onclick=closeDrawer; els.shareBtn.onclick=share; els.singleModeBtn.onclick=()=>setMode('single'); els.scrollModeBtn.onclick=()=>setMode('scroll'); els.widthModeBtn.onclick=()=>setZoom('width'); els.fitModeBtn.onclick=()=>setZoom('fit'); els.resetBtn.onclick=reset;
    els.continueBtn.onclick=()=>{const p=state().page||1;closeDrawer();showPage(p)};
    els.startBtn.onclick=()=>{localStorage.setItem(FIRST_VISIT_KEY,'1');els.welcome.classList.add('hidden');showPage(1)};
    els.welcomeContinueBtn.onclick=()=>{localStorage.setItem(FIRST_VISIT_KEY,'1');els.welcome.classList.add('hidden');showPage(state().page||1)};
    document.addEventListener('keydown',e=>{ if(els.drawer.classList.contains('open')){if(e.key==='Escape')closeDrawer();return;} if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();go(1)} if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();go(-1)} if(e.key==='Home'){e.preventDefault();showPage(1)} if(e.key==='End'){e.preventDefault();showPage(TOTAL)} });
    let touchX=null,touchY=null; els.singleMode.addEventListener('touchstart',e=>{const t=e.changedTouches[0];touchX=t.clientX;touchY=t.clientY},{passive:true}); els.singleMode.addEventListener('touchend',e=>{if(touchX===null)return;const t=e.changedTouches[0];const dx=t.clientX-touchX,dy=t.clientY-touchY;touchX=touchY=null;if(Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.2) go(dx<0?1:-1)},{passive:true});
    window.addEventListener('hashchange',()=>{const p=parseHash();if(p&&p!==currentPage)showPage(p)});
  }

  async function init(){
    bind(); buildGrid(); const s=state(); mode=s.mode==='scroll'?'scroll':'single'; zoom=s.zoom==='fit'?'fit':'width'; setZoom(zoom); updateDrawerProgress();
    const deep=parseHash(); currentPage=deep || clamp(s.page||1,1,TOTAL);
    els.singleModeBtn.classList.toggle('active',mode==='single'); els.scrollModeBtn.classList.toggle('active',mode==='scroll'); els.singleMode.classList.toggle('hidden',mode!=='single'); els.scrollMode.classList.toggle('hidden',mode!=='scroll');
    const firstVisit=!localStorage.getItem(FIRST_VISIT_KEY) && !deep && !(s.page>1); if(firstVisit) els.welcome.classList.remove('hidden'); else els.welcome.classList.add('hidden');
    if(mode==='scroll'){buildScroll();requestAnimationFrame(()=>scrollToScrollPage(currentPage,false));} else await showPage(currentPage,{scrollTop:false}); updateChrome();
  }
  init();
})();
