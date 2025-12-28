(function () {
    const saved = localStorage.getItem('theme');
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    const theme = saved || (prefersLight ? 'light' : 'dark');
    document.documentElement.dataset.theme = theme;
    const darkLink = document.getElementById('hljs-dark');
    const lightLink = document.getElementById('hljs-light');
    if (darkLink) darkLink.disabled = theme !== 'dark';
    if (lightLink) lightLink.disabled = theme !== 'light';
})();

function copyCode(button) {
    const code = button.nextElementSibling.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        const original = button.textContent;
        button.textContent = 'Copied!';
        setTimeout(() => button.textContent = original, 2000);
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
            const darkLink = document.getElementById('hljs-dark');
            const lightLink = document.getElementById('hljs-light');
            if (darkLink) darkLink.disabled = newTheme !== 'dark';
            if (lightLink) lightLink.disabled = newTheme !== 'light';
            hljs.highlightAll();
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
            const matchesSearch = !query || title.includes(query);
            const matchesTags = selectedTags.length === 0 || selectedTags.some(t => tags.includes(t));
            post.style.display = (matchesSearch && matchesTags) ? '' : 'none';
        });
    };

    if (searchInput) searchInput.addEventListener('input', filter);
    tagFilters.forEach(cb => cb.addEventListener('change', filter));
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
        document.getElementById('lightbox-media').src = element.src;
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
    }

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateTransform();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            const media = document.getElementById('lightbox-media');
            if (media) {
                media.style.cursor = currentZoom > 1 ? 'grab' : 'default';
            }
        }
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
});
