(function () {
    const saved = localStorage.getItem('theme');
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    const theme = saved || (prefersLight ? 'light' : 'dark');
    document.documentElement.dataset.theme = theme;
})();

function showCopyToast(message) {
    let toast = document.getElementById('copy-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'copy-toast';
        toast.className = 'copy-toast';
        document.body.appendChild(toast);
    }
    toast.textContent = message || 'Copied!';
    toast.classList.add('visible');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('visible'), 1800);
}

function copyWithToast(text, message, onDone) {
    const finish = () => {
        showCopyToast(message || 'Copied!');
        if (onDone) onDone();
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(finish, () => {
            fallbackCopy(text);
            finish();
        });
    } else {
        fallbackCopy(text);
        finish();
    }
}

function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) { }
    ta.remove();
}

window.copyWithToast = copyWithToast;

function copyCode(button) {
    const wrapper = button.closest('.code-block-wrapper');
    const codeEl = wrapper ? wrapper.querySelector('pre code') : button.nextElementSibling.querySelector('code');
    if (!codeEl) return;
    const code = codeEl.textContent;
    const original = button.textContent;
    copyWithToast(code, 'Code copied!', () => {
        button.textContent = 'Copied';
        button.classList.add('copied');
        setTimeout(() => {
            button.textContent = original;
            button.classList.remove('copied');
        }, 1600);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const root = document.documentElement;
    const inputs = document.querySelectorAll('.switch input');

    const currentTheme = root.dataset.theme || 'dark';
    inputs.forEach(i => i.checked = currentTheme === 'light');

    inputs.forEach(input => {
        input.addEventListener('change', () => {
            const newTheme = input.checked ? 'light' : 'dark';
            root.dataset.theme = newTheme;
            localStorage.setItem('theme', newTheme);
        });
    });

    const searchInput = document.getElementById('search-input');
    const tagFilters = document.querySelectorAll('.tag-filter');
    const posts = document.querySelectorAll('.post-item');

    const filter = () => {
        const query = (searchInput?.value || '').toLowerCase();
        const selectedTags = Array.from(tagFilters)
            .filter(c => c.checked)
            .map(c => c.dataset.tag.toLowerCase());

        posts.forEach(post => {
            const title = post.querySelector('.post-info a')?.textContent.toLowerCase() || '';
            const tags = (post.dataset.tags || '').toLowerCase().split(' ');
            const matchesSearch = !query || title.includes(query) || tags.some(t => t.includes(query));
            const matchesTags = selectedTags.length === 0 || selectedTags.some(t => tags.includes(t));
            post.style.display = (matchesSearch && matchesTags) ? '' : 'none';
        });

        document.querySelectorAll('.tag-btn').forEach(btn => {
            btn.classList.toggle('active', selectedTags.includes((btn.dataset.tag || '').toLowerCase()));
        });
    };

    if (searchInput) {
        searchInput.addEventListener('input', filter);
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                const tag = document.activeElement?.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
                e.preventDefault();
                searchInput.focus();
                searchInput.select();
            } else if (e.key === 'Escape' && document.activeElement === searchInput) {
                if (searchInput.value) {
                    searchInput.value = '';
                    filter();
                } else {
                    searchInput.blur();
                }
            }
        });
    }
    tagFilters.forEach(cb => cb.addEventListener('change', filter));
    document.querySelectorAll('.tag-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.dataset.tag;
            const cb = Array.from(tagFilters).find(c => c.dataset.tag === t);
            if (cb) {
                cb.checked = !cb.checked;
                filter();
            }
        });
    });

    // Reading progress bar
    const progressBar = document.getElementById('reading-progress');
    if (progressBar) {
        let ticking = false;
        const updateProgress = () => {
            const docH = document.documentElement.scrollHeight - window.innerHeight;
            const pct = docH > 0 ? Math.min(100, Math.max(0, (window.scrollY / docH) * 100)) : 0;
            progressBar.style.width = pct + '%';
            ticking = false;
        };
        window.addEventListener('scroll', () => {
            if (!ticking) {
                ticking = true;
                requestAnimationFrame(updateProgress);
            }
        }, { passive: true });
        updateProgress();
    }

    // Heading anchor copy-to-clipboard
    document.querySelectorAll('.heading-anchor').forEach(anchor => {
        anchor.addEventListener('click', () => {
            const href = anchor.getAttribute('href');
            if (!href) return;
            const url = window.location.origin + window.location.pathname + href;
            copyWithToast(url, 'Link copied!', () => {
                anchor.classList.add('copied');
                anchor.textContent = '✓';
                setTimeout(() => {
                    anchor.classList.remove('copied');
                    anchor.textContent = '#';
                }, 1400);
            });
        });
    });

    // Table of contents scroll-spy
    const toc = document.getElementById('post-toc');
    if (toc) {
        const tocLinks = Array.from(toc.querySelectorAll('.toc-link'));
        const headingEls = tocLinks
            .map(l => document.getElementById(l.getAttribute('href').slice(1)))
            .filter(Boolean);
        if (headingEls.length > 0) {
            let activeId = headingEls[0].id;
            const setActive = (id) => {
                activeId = id;
                tocLinks.forEach(l => {
                    l.classList.toggle('active', l.getAttribute('href') === '#' + id);
                });
            };
            let spyTick = false;
            const onScrollSpy = () => {
                const y = window.scrollY + 110;
                let current = headingEls[0].id;
                for (let i = 0; i < headingEls.length; i++) {
                    if (headingEls[i].offsetTop <= y) current = headingEls[i].id;
                    else break;
                }
                if (current !== activeId) setActive(current);
                spyTick = false;
            };
            window.addEventListener('scroll', () => {
                if (!spyTick) {
                    spyTick = true;
                    requestAnimationFrame(onScrollSpy);
                }
            }, { passive: true });
            setActive(activeId);
        }
    }
});

// lightbox for image/video viewing

let currentZoom = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

function openLightbox(element) {
    const lightbox = document.getElementById('lightbox');
    if (!lightbox) return;
    
    const media = document.getElementById('lightbox-media');
    
    currentZoom = 1;
    translateX = 0;
    translateY = 0;
    
    if (element.tagName === 'VIDEO') {
        const video = document.createElement('video');
        video.src = element.src;
        video.controls = true;
        video.className = 'lightbox-media';
        video.id = 'lightbox-media';
        media.replaceWith(video);
    } else {
        if (media.tagName === 'VIDEO') {
            const img = document.createElement('img');
            img.className = 'lightbox-media';
            img.id = 'lightbox-media';
            media.replaceWith(img);
        }
        const lbImg = document.getElementById('lightbox-media');
        lbImg.src = element.src;
        lbImg.classList.toggle('theme-adapt', element.classList.contains('theme-adapt'));
        lbImg.classList.toggle('theme-adapt-dark', element.classList.contains('theme-adapt-dark'));
    }
    
    lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';
    updateTransform();
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    if (!lightbox) return;
    
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
    
    const media = document.getElementById('lightbox-media');
    if (media.tagName === 'VIDEO') {
        media.pause();
    }
}

function zoomIn() {
    currentZoom = Math.min(currentZoom + 0.25, 5);
    updateTransform();
}

function zoomOut() {
    currentZoom = Math.max(currentZoom - 0.25, 0.5);
    updateTransform();
}

function resetZoom() {
    currentZoom = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
}

function updateTransform() {
    const media = document.getElementById('lightbox-media');
    if (!media) return;
    
    media.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentZoom})`;
    const zoomLevel = document.getElementById('zoom-level');
    if (zoomLevel) {
        zoomLevel.textContent = `${Math.round(currentZoom * 100)}%`;
    }
    media.style.cursor = currentZoom > 1 ? 'grab' : 'default';
}

document.addEventListener('DOMContentLoaded', () => {
    const lightbox = document.getElementById('lightbox');
    if (lightbox) {
        lightbox.addEventListener('click', (e) => {
            if (e.target.id === 'lightbox') {
                closeLightbox();
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (!lightbox || !lightbox.classList.contains('active')) return;
        
        if (e.key === 'Escape') closeLightbox();
        else if (e.key === '+' || e.key === '=') zoomIn();
        else if (e.key === '-') zoomOut();
        else if (e.key === '0') resetZoom();
    });

    const mediaEl = document.getElementById('lightbox-media');
    if (mediaEl) {
        mediaEl.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) {
                zoomIn();
            } else {
                zoomOut();
            }
        });

        mediaEl.addEventListener('mousedown', (e) => {
            if (currentZoom <= 1) return;
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            mediaEl.style.cursor = 'grabbing';
        });

        mediaEl.addEventListener('touchstart', (e) => {
            if (currentZoom <= 1 || e.touches.length !== 1) return;
            isDragging = true;
            startX = e.touches[0].clientX - translateX;
            startY = e.touches[0].clientY - translateY;
        }, { passive: true });
    }

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateTransform();
    });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        translateX = e.touches[0].clientX - startX;
        translateY = e.touches[0].clientY - startY;
        updateTransform();
    }, { passive: true });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            const media = document.getElementById('lightbox-media');
            if (media) {
                media.style.cursor = currentZoom > 1 ? 'grab' : 'default';
            }
        }
    });

    document.addEventListener('touchend', () => {
        isDragging = false;
    });

    const images = document.querySelectorAll('main img, main video');
    images.forEach(img => {
        img.addEventListener('click', () => openLightbox(img));
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const burger = document.querySelector('.burger-menu');
    const nav = document.querySelector('nav');
    
    if (burger) {
        burger.addEventListener('click', () => {
            burger.classList.toggle('active');
            nav.classList.toggle('active');
        });
        document.addEventListener('click', (e) => {
            if (!burger.contains(e.target) && !nav.contains(e.target) && nav.classList.contains('active')) {
                burger.classList.remove('active');
                nav.classList.remove('active');
            }
        });
    }

    const filterToggle = document.querySelector('.filter-toggle');
    const rightSidebar = document.querySelector('.right-sidebar');
    
    if (filterToggle && rightSidebar) {
        filterToggle.addEventListener('click', () => {
            rightSidebar.classList.toggle('active');
        });
        document.addEventListener('click', (e) => {
            if (!filterToggle.contains(e.target) && !rightSidebar.contains(e.target) && rightSidebar.classList.contains('active')) {
                rightSidebar.classList.remove('active');
            }
        });
    }

    var pwTrigger = document.getElementById('past-work-trigger');
    var pwModal = document.getElementById('past-work-modal');
    if (pwTrigger && pwModal) {
        var pwBackdrop = pwModal.querySelector('.pw-modal-backdrop');
        var pwClose = pwModal.querySelector('.pw-modal-close');
        pwTrigger.addEventListener('click', function (e) {
            e.preventDefault();
            pwModal.classList.add('active');
            document.body.classList.add('modal-open');
            document.body.style.overflow = 'hidden';
        });
        var closePwModal = function () {
            pwModal.classList.remove('active');
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
        };
        pwBackdrop.addEventListener('click', closePwModal);
        pwClose.addEventListener('click', closePwModal);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && pwModal.classList.contains('active')) {
                e.stopImmediatePropagation();
                closePwModal();
            }
        });
    }

    const infoBtn = document.querySelector('.post-info-btn');
    const postStats = document.querySelector('.post-stats');
    if (infoBtn && postStats) {
        let tooltip = null;
        infoBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (tooltip) {
                tooltip.remove();
                tooltip = null;
                return;
            }
            tooltip = document.createElement('div');
            tooltip.className = 'post-info-tooltip';
            tooltip.textContent = postStats.textContent;
            infoBtn.parentElement.appendChild(tooltip);
            
            document.addEventListener('click', function closeTooltip() {
                if (tooltip) {
                    tooltip.remove();
                    tooltip = null;
                }
                document.removeEventListener('click', closeTooltip);
            });
        });
    }

    // Co-author live hover preview card (desktop hover pointers only)
    const authorLinks = document.querySelectorAll('.coauthor-link');
    if (authorLinks.length > 0 && window.matchMedia('(hover: hover)').matches) {
        const card = document.createElement('div');
        card.className = 'author-preview-card';
        card.innerHTML =
            '<div class="ap-shot-wrap"></div>' +
            '<div class="ap-meta">' +
              '<div class="ap-top">' +
                '<span class="ap-name"></span>' +
                '<span class="ap-ext">&#8599;</span>' +
              '</div>' +
              '<div class="ap-sub">' +
                '<span class="ap-affil"></span>' +
                '<span class="ap-host"></span>' +
              '</div>' +
            '</div>';
        document.body.appendChild(card);

        const elName = card.querySelector('.ap-name');
        const elAffil = card.querySelector('.ap-affil');
        const elHost = card.querySelector('.ap-host');
        const elShotWrap = card.querySelector('.ap-shot-wrap');
        let hideTimer = null;
        const frameCache = new Map();

        const positionCard = (link) => {
            const rect = link.getBoundingClientRect();
            const cw = card.offsetWidth || 300;
            const ch = card.offsetHeight || 240;
            let left = rect.left + rect.width / 2 - cw / 2;
            left = Math.max(12, Math.min(window.innerWidth - cw - 12, left));
            let top = rect.top - ch - 10;
            if (top < 12) top = rect.bottom + 10;
            if (top + ch > window.innerHeight - 12) top = Math.max(12, window.innerHeight - ch - 12);
            card.style.left = left + 'px';
            card.style.top = top + 'px';
        };

        const showCard = (link) => {
            clearTimeout(hideTimer);
            elName.textContent = link.dataset.name || link.textContent;
            const affil = link.dataset.affil || '';
            elAffil.textContent = affil;
            elAffil.style.display = affil ? 'inline' : 'none';
            elHost.textContent = link.dataset.host || '';

            const url = link.href;
            let entry = frameCache.get(url);
            if (!entry) {
                const iframe = document.createElement('iframe');
                iframe.className = 'ap-frame';
                iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
                iframe.setAttribute('referrerpolicy', 'no-referrer');
                iframe.setAttribute('tabindex', '-1');
                iframe.setAttribute('aria-hidden', 'true');
                iframe.style.opacity = '0';
                entry = { iframe, loaded: false };
                iframe.addEventListener('load', () => {
                    entry.loaded = true;
                    iframe.style.opacity = '1';
                }, { once: true });
                iframe.src = url;
                elShotWrap.appendChild(iframe);
                frameCache.set(url, entry);
            }

            frameCache.forEach((item, key) => {
                item.iframe.style.display = key === url ? 'block' : 'none';
            });
            entry.iframe.style.opacity = entry.loaded ? '1' : '0';

            card.classList.add('visible');
            positionCard(link);
        };

        const hideCard = () => {
            hideTimer = setTimeout(() => card.classList.remove('visible'), 120);
        };

        authorLinks.forEach(link => {
            link.addEventListener('mouseenter', () => showCard(link));
            link.addEventListener('mouseleave', hideCard);
            link.addEventListener('focus', () => showCard(link));
            link.addEventListener('blur', hideCard);
        });
    }

    // Command palette (Cmd+K / Ctrl+K)
    const cmdTrigger = document.getElementById('cmd-trigger');
    let cmdEl = null;
    let cmdPosts = null;
    let cmdItems = [];
    let cmdIdx = 0;

    const buildBaseActions = () => {
        const actions = [
            { label: 'Home', kind: 'NAV', run: () => { window.location.href = '/'; } },
            { label: 'Posts', kind: 'NAV', run: () => { window.location.href = '/posts/'; } },
            {
                label: 'Papers, experience & talks',
                kind: 'VIEW',
                run: () => {
                    const trigger = document.getElementById('past-work-trigger');
                    if (trigger) trigger.click();
                    else window.location.href = '/#work';
                }
            },
            {
                label: 'Toggle light / dark theme',
                kind: 'SYS',
                run: () => {
                    const sw = document.querySelector('.switch input');
                    if (sw) { sw.checked = !sw.checked; sw.dispatchEvent(new Event('change')); }
                }
            },
            {
                label: 'Toggle Particle Life simulation',
                kind: 'SYS',
                run: () => {
                    const btn = document.getElementById('immersive-toggle');
                    if (btn) btn.click();
                }
            }
        ];
        if (cmdPosts) {
            cmdPosts.forEach(p => {
                actions.push({
                    label: p.title,
                    meta: p.date,
                    kind: 'POST',
                    run: () => { window.location.href = p.url; }
                });
            });
        }
        return actions;
    };

    const ensureCmdPalette = () => {
        if (cmdEl) return cmdEl;
        cmdEl = document.createElement('div');
        cmdEl.className = 'cmd-palette';
        cmdEl.innerHTML =
            '<div class="cmd-backdrop"></div>' +
            '<div class="cmd-palette-box" role="dialog" aria-label="Command palette">' +
              '<div class="cmd-input-row">' +
                '<span class="cmd-prompt">&gt;</span>' +
                '<input type="text" class="cmd-input" placeholder="Jump to page, post, or command..." autocomplete="off" spellcheck="false">' +
                '<kbd class="cmd-esc">ESC</kbd>' +
              '</div>' +
              '<ul class="cmd-list"></ul>' +
            '</div>';
        document.body.appendChild(cmdEl);

        const input = cmdEl.querySelector('.cmd-input');
        const list = cmdEl.querySelector('.cmd-list');
        const backdrop = cmdEl.querySelector('.cmd-backdrop');

        const renderList = () => {
            const q = input.value.trim().toLowerCase();
            const all = buildBaseActions();
            cmdItems = q
                ? all.filter(it => (it.label + ' ' + (it.meta || '') + ' ' + it.kind).toLowerCase().includes(q))
                : all;
            if (cmdIdx >= cmdItems.length) cmdIdx = 0;
            list.innerHTML = '';
            if (cmdItems.length === 0) {
                const empty = document.createElement('li');
                empty.className = 'cmd-empty';
                empty.textContent = 'No matching commands.';
                list.appendChild(empty);
                return;
            }
            cmdItems.forEach((it, idx) => {
                const li = document.createElement('li');
                li.className = 'cmd-item' + (idx === cmdIdx ? ' active' : '');
                li.innerHTML =
                    '<span class="cmd-kind">' + it.kind + '</span>' +
                    '<span class="cmd-label"></span>' +
                    (it.meta ? '<span class="cmd-meta">' + it.meta + '</span>' : '');
                li.querySelector('.cmd-label').textContent = it.label;
                li.addEventListener('mouseenter', () => {
                    cmdIdx = idx;
                    list.querySelectorAll('.cmd-item').forEach((n, i) => n.classList.toggle('active', i === cmdIdx));
                });
                li.addEventListener('click', () => {
                    closeCmd();
                    it.run();
                });
                list.appendChild(li);
            });
        };

        input.addEventListener('input', () => { cmdIdx = 0; renderList(); });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (cmdItems.length) { cmdIdx = (cmdIdx + 1) % cmdItems.length; renderList(); }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (cmdItems.length) { cmdIdx = (cmdIdx - 1 + cmdItems.length) % cmdItems.length; renderList(); }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (cmdItems[cmdIdx]) { const fn = cmdItems[cmdIdx].run; closeCmd(); fn(); }
            } else if (e.key === 'Escape') {
                closeCmd();
            }
        });
        backdrop.addEventListener('click', closeCmd);
        cmdEl._render = renderList;
        return cmdEl;
    };

    const openCmd = () => {
        const el = ensureCmdPalette();
        el.classList.add('active');
        const input = el.querySelector('.cmd-input');
        input.value = '';
        cmdIdx = 0;
        el._render();
        setTimeout(() => input.focus(), 10);
        if (cmdPosts === null) {
            fetch('/posts.json')
                .then(r => r.ok ? r.json() : [])
                .then(data => { cmdPosts = Array.isArray(data) ? data : []; el._render(); })
                .catch(() => { cmdPosts = []; });
        }
    };

    const closeCmd = () => {
        if (cmdEl) cmdEl.classList.remove('active');
    };

    if (cmdTrigger) cmdTrigger.addEventListener('click', openCmd);
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            if (cmdEl && cmdEl.classList.contains('active')) closeCmd();
            else openCmd();
        }
    });

    // Support /#work deep-link from Command Palette
    if (window.location.hash === '#work') {
        const trigger = document.getElementById('past-work-trigger');
        if (trigger) trigger.click();
    }
});
