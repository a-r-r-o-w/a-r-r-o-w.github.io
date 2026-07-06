(function () {
    var canvas = document.getElementById('particle-life');
    if (!canvas) {
        return;
    }

    var gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false, antialias: false });
    if (!gl) {
        gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false });
    }
    if (!gl) {
        return;
    }
    var isWebGL2 = !!gl.createTransformFeedback;

    var NUM_SPECIES = 6, NUM_PARTICLES = 12000;
    var SESSION_DURATION = 60000, FADE_DURATION = 3500;
    var RMAX = 80.0, FRICTION = 0.05, DT = 0.004, FORCE_SCALE = 12.0;
    var REPULSION_SCALE = 1.0, VEL_DAMPING = 1.0, WALL_MODE = 1;
    var immersive = false, immersiveT = 0.0;
    var W, H, dpr, contentL = 0, contentR = 0, rectTimer = 0;
    var seed = (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0;
    var rngState = seed;
    var worker = null, frameData = null, frameN = 0, pendingBuf = null;
    var sessionStart, prevTime, running = false, animId = null;
    var started = false;
    var controlsPanel = null, speciesPanel = null, immersiveGlow = 0.75;
    var editorMode = 'simple';
    var currentAttractions = null, currentMinDist = null;

    var FLOATS_PER_PARTICLE = 7;
    var STRIDE_BYTES = 8 * 4;

    var lifecycle = {
        enabled: false, reproRate: 0.5, mutationRate: 0.2,
        lifespan: 3000, energyGain: 0.5, energyCost: 0.3, predationStr: 0.5
    };

    var darkPalette = [
        [1.00, 0.30, 0.35], [0.20, 0.80, 1.00], [0.40, 1.00, 0.40],
        [1.00, 0.85, 0.15], [0.85, 0.35, 1.00], [1.00, 0.55, 0.10],
        [0.10, 1.00, 0.80], [1.00, 0.40, 0.75], [0.50, 0.70, 1.00],
        [0.80, 1.00, 0.20], [1.00, 0.65, 0.50], [0.60, 0.40, 1.00],
    ];
    var lightPalette = [
        [0.85, 0.10, 0.15], [0.05, 0.50, 0.80], [0.10, 0.65, 0.10],
        [0.75, 0.60, 0.00], [0.60, 0.10, 0.75], [0.80, 0.35, 0.00],
        [0.00, 0.65, 0.55], [0.80, 0.15, 0.50], [0.20, 0.40, 0.80],
        [0.50, 0.65, 0.00], [0.80, 0.40, 0.25], [0.35, 0.15, 0.75],
    ];
    var speciesNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'];
    var speciesShapes = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5];

    var plIconSvg = '<svg class="pl-icon" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">' +
        '<circle cx="4" cy="4" r="1.8" opacity="0.9"/>' +
        '<circle cx="12" cy="5" r="1.4" opacity="0.7"/>' +
        '<circle cx="7" cy="12" r="1.6" opacity="0.8"/>' +
        '<circle cx="10" cy="10" r="1.0" opacity="0.5"/>' +
        '<circle cx="3" cy="9" r="0.8" opacity="0.4"/>' +
        '<circle cx="13" cy="13" r="0.7" opacity="0.35"/>' +
        '</svg>';

    function xorshift32() {
        rngState ^= rngState << 13;
        rngState ^= rngState >>> 17;
        rngState ^= rngState << 5;
        return (rngState >>> 0) / 0xFFFFFFFF;
    }

    function measureContent() {
        var el = document.querySelector('.container') || document.querySelector('.page-wrapper');
        if (el) {
            var r = el.getBoundingClientRect();
            contentL = r.left;
            contentR = r.right;
        }
        else {
            contentL = W * 0.25;
            contentR = W * 0.75;
        }
    }

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        W = window.innerWidth;
        H = window.innerHeight;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        gl.viewport(0, 0, canvas.width, canvas.height);
        measureContent();
    }

    function generateRules() {
        var n = NUM_SPECIES;
        var a = new Float32Array(n * n);
        var m = new Float32Array(n * n);
        var i, j, idx;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                idx = i * n + j;
                a[idx] = (xorshift32() * 2.0 - 1.0) * 0.8;
                m[idx] = 0.15 + xorshift32() * 0.25;
            }
        }
        return { attractions: a, minDist: m };
    }

    function getZones() {
        var gap = 40;
        if (immersive) {
            return { left: W * 0.5, right: W * 0.5 };
        }
        return {
            left: Math.max(0, contentL - gap),
            right: Math.min(W, contentR + gap)
        };
    }

    function initWorker() {
        if (worker) {
            worker.terminate();
        }
        var rules = generateRules();
        currentAttractions = rules.attractions;
        currentMinDist = rules.minDist;
        var z = getZones();

        worker = new Worker('/assets/particle-worker.js');
        worker.onmessage = function (e) {
            var d = e.data;
            if (d.type === 'frame') {
                var src = new Float32Array(d.buf);
                if (!frameData || frameData.length < src.length) {
                    frameData = new Float32Array(src.length);
                }
                frameData.set(src);
                frameN = d.n;
                pendingBuf = d.buf;
            }
        };
        worker.postMessage({
            type: 'init',
            config: {
                numParticles: NUM_PARTICLES,
                numSpecies: NUM_SPECIES,
                worldW: W, worldH: H,
                rmax: RMAX, friction: FRICTION, dt: DT, forceScale: FORCE_SCALE,
                attractions: currentAttractions, minDist: currentMinDist,
                leftZoneEnd: z.left, rightZoneStart: z.right,
                seed: seed, repulsionScale: REPULSION_SCALE,
                velDamping: VEL_DAMPING, wallMode: WALL_MODE,
                lifecycle: lifecycle,
            }
        });

        if (speciesPanel && editorMode === 'advanced') {
            rebuildSpeciesPanel();
        }
    }

    function resetSession() {
        seed = (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0;
        rngState = seed;
        sessionStart = performance.now();
        prevTime = sessionStart;
        measureContent();
        initWorker();
    }

    function sendUpdate(key, val) {
        if (worker) {
            var msg = { type: 'update' };
            msg[key] = val;
            worker.postMessage(msg);
        }
    }

    function paletteToHex(c) {
        var r = Math.round(c[0] * 255).toString(16).padStart(2, '0');
        var g = Math.round(c[1] * 255).toString(16).padStart(2, '0');
        var b = Math.round(c[2] * 255).toString(16).padStart(2, '0');
        return '#' + r + g + b;
    }

    function exportConfig() {
        var n = NUM_SPECIES;
        var att = [], md = [];
        for (var i = 0; i < n * n; i++) {
            att.push(Math.round(currentAttractions[i] * 1000) / 1000);
            md.push(Math.round(currentMinDist[i] * 1000) / 1000);
        }
        var cfg = {
            version: 2, numSpecies: n, numParticles: NUM_PARTICLES,
            rmax: RMAX, friction: FRICTION, dt: DT, forceScale: FORCE_SCALE,
            repulsionScale: REPULSION_SCALE, velDamping: VEL_DAMPING,
            wallMode: WALL_MODE, glow: immersiveGlow,
            attractions: att, minDist: md, lifecycle: lifecycle,
        };
        var blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'particle-life-config.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function importConfig(file) {
        var reader = new FileReader();
        reader.onload = function (e) {
            try {
                var cfg = JSON.parse(e.target.result);
                if (!cfg.attractions || !cfg.minDist) {
                    return;
                }
                NUM_SPECIES = cfg.numSpecies || 6;
                NUM_PARTICLES = cfg.numParticles || 12000;
                RMAX = cfg.rmax || 80;
                FRICTION = cfg.friction || 0.05;
                DT = cfg.dt || 0.004;
                FORCE_SCALE = cfg.forceScale || 12;
                REPULSION_SCALE = cfg.repulsionScale || 1;
                VEL_DAMPING = cfg.velDamping || 1;
                WALL_MODE = cfg.wallMode !== undefined ? cfg.wallMode : 1;
                immersiveGlow = cfg.glow || 0.75;
                if (cfg.lifecycle) {
                    lifecycle = cfg.lifecycle;
                }
                currentAttractions = new Float32Array(cfg.attractions);
                currentMinDist = new Float32Array(cfg.minDist);
                reallocVertexData();
                seed = (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0;
                rngState = seed;
                sessionStart = performance.now();
                measureContent();
                if (worker) {
                    worker.terminate();
                }
                var z = getZones();
                worker = new Worker('/assets/particle-worker.js');
                worker.onmessage = function (ev) {
                    var d = ev.data;
                    if (d.type === 'frame') {
                        var src = new Float32Array(d.buf);
                        if (!frameData || frameData.length < src.length) {
                            frameData = new Float32Array(src.length);
                        }
                        frameData.set(src);
                        frameN = d.n;
                        pendingBuf = d.buf;
                    }
                };
                worker.postMessage({
                    type: 'init',
                    config: {
                        numParticles: NUM_PARTICLES, numSpecies: NUM_SPECIES,
                        worldW: W, worldH: H,
                        rmax: RMAX, friction: FRICTION, dt: DT, forceScale: FORCE_SCALE,
                        attractions: currentAttractions, minDist: currentMinDist,
                        leftZoneEnd: z.left, rightZoneStart: z.right,
                        seed: seed, repulsionScale: REPULSION_SCALE,
                        velDamping: VEL_DAMPING, wallMode: WALL_MODE,
                        lifecycle: lifecycle,
                    }
                });
                removeControls();
                removeSpeciesPanel();
                buildControls();
                if (editorMode === 'advanced') {
                    buildSpeciesPanel();
                }
            }
            catch (err) { }
        };
        reader.readAsText(file);
    }

    var vsSrc = isWebGL2
        ? '#version 300 es\nin vec2 aPos;\nin vec4 aColor;\nin float aSize;\nin float aShape;\nuniform vec2 uRes;\nout vec4 vColor;\nout float vShape;\nvoid main() {\n    vec2 c = (aPos / uRes) * 2.0 - 1.0;\n    c.y = -c.y;\n    gl_Position = vec4(c, 0.0, 1.0);\n    gl_PointSize = aSize;\n    vColor = aColor;\n    vShape = aShape;\n}\n'
        : 'attribute vec2 aPos;\nattribute vec4 aColor;\nattribute float aSize;\nattribute float aShape;\nuniform vec2 uRes;\nvarying vec4 vColor;\nvarying float vShape;\nvoid main() {\n    vec2 c = (aPos / uRes) * 2.0 - 1.0;\n    c.y = -c.y;\n    gl_Position = vec4(c, 0.0, 1.0);\n    gl_PointSize = aSize;\n    vColor = aColor;\n    vShape = aShape;\n}\n';

    var fsSrc = isWebGL2
        ? '#version 300 es\nprecision mediump float;\nin vec4 vColor;\nin float vShape;\nout vec4 fc;\nvoid main() {\n    vec2 d = gl_PointCoord - 0.5;\n    float r2 = dot(d, d);\n    int s = int(vShape + 0.5);\n    float mask = 0.0;\n    if (s == 0) {\n        mask = smoothstep(0.25, 0.15, r2);\n    } else if (s == 1) {\n        float ax = abs(d.x), ay = abs(d.y);\n        mask = step(ax + ay, 0.42) * smoothstep(0.42, 0.35, ax + ay);\n    } else if (s == 2) {\n        float ax = abs(d.x), ay = abs(d.y);\n        mask = step(max(ax, ay), 0.38) * smoothstep(0.38, 0.30, max(ax, ay));\n    } else if (s == 3) {\n        float tx = d.x * 1.8, ty = d.y + 0.15;\n        float tri = max(abs(tx) - 0.5 + ty, -ty - 0.30);\n        mask = smoothstep(0.02, -0.02, tri);\n    } else if (s == 4) {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float star = r - 0.25 - 0.12 * cos(a * 5.0);\n        mask = smoothstep(0.02, -0.02, star);\n    } else {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float hex = r - 0.35 + 0.06 * cos(a * 6.0);\n        mask = smoothstep(0.02, -0.02, hex);\n    }\n    if (mask < 0.01) discard;\n    fc = vec4(vColor.rgb, vColor.a * mask);\n}\n'
        : 'precision mediump float;\nvarying vec4 vColor;\nvarying float vShape;\nvoid main() {\n    vec2 d = gl_PointCoord - 0.5;\n    float r2 = dot(d, d);\n    int s = int(vShape + 0.5);\n    float mask = 0.0;\n    if (s == 0) {\n        mask = smoothstep(0.25, 0.15, r2);\n    } else if (s == 1) {\n        float ax = abs(d.x), ay = abs(d.y);\n        mask = step(ax + ay, 0.42) * smoothstep(0.42, 0.35, ax + ay);\n    } else if (s == 2) {\n        float ax = abs(d.x), ay = abs(d.y);\n        mask = step(max(ax, ay), 0.38) * smoothstep(0.38, 0.30, max(ax, ay));\n    } else if (s == 3) {\n        float tx = d.x * 1.8, ty = d.y + 0.15;\n        float tri = max(abs(tx) - 0.5 + ty, -ty - 0.30);\n        mask = smoothstep(0.02, -0.02, tri);\n    } else if (s == 4) {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float star = r - 0.25 - 0.12 * cos(a * 5.0);\n        mask = smoothstep(0.02, -0.02, star);\n    } else {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float hex = r - 0.35 + 0.06 * cos(a * 6.0);\n        mask = smoothstep(0.02, -0.02, hex);\n    }\n    if (mask < 0.01) discard;\n    gl_FragColor = vec4(vColor.rgb, vColor.a * mask);\n}\n';

    function compileShader(type, source) {
        var s = gl.createShader(type);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            gl.deleteShader(s);
            return null;
        }
        return s;
    }

    var vs = compileShader(gl.VERTEX_SHADER, vsSrc);
    var fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) {
        return;
    }
    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        return;
    }
    gl.useProgram(prog);

    var aPosLoc = gl.getAttribLocation(prog, 'aPos');
    var aColorLoc = gl.getAttribLocation(prog, 'aColor');
    var aSizeLoc = gl.getAttribLocation(prog, 'aSize');
    var aShapeLoc = gl.getAttribLocation(prog, 'aShape');
    var uResLoc = gl.getUniformLocation(prog, 'uRes');

    var maxVerts = 40000;
    var vertexData = new Float32Array(maxVerts * 8);
    var vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, STRIDE_BYTES, 0);
    gl.enableVertexAttribArray(aColorLoc);
    gl.vertexAttribPointer(aColorLoc, 4, gl.FLOAT, false, STRIDE_BYTES, 8);
    gl.enableVertexAttribArray(aSizeLoc);
    gl.vertexAttribPointer(aSizeLoc, 1, gl.FLOAT, false, STRIDE_BYTES, 24);
    if (aShapeLoc >= 0) {
        gl.enableVertexAttribArray(aShapeLoc);
        gl.vertexAttribPointer(aShapeLoc, 1, gl.FLOAT, false, STRIDE_BYTES, 28);
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    function reallocVertexData() {
        var needed = Math.max(NUM_PARTICLES, 40000) * 8;
        if (vertexData.length < needed) {
            vertexData = new Float32Array(needed);
            gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
            gl.bufferData(gl.ARRAY_BUFFER, vertexData.byteLength, gl.DYNAMIC_DRAW);
        }
    }

    resize();
    window.addEventListener('resize', resize);
    sessionStart = performance.now();
    prevTime = sessionStart;

    function getPalette() {
        if (document.documentElement.dataset.theme === 'light') {
            return lightPalette;
        }
        return darkPalette;
    }

    function render(now) {
        if (!running) {
            return;
        }
        animId = requestAnimationFrame(render);

        if (pendingBuf) {
            var b = pendingBuf;
            pendingBuf = null;
            worker.postMessage({ type: 'step', buf: b }, [b]);
        }

        var elapsed = now - sessionStart;
        var globalAlpha = 1.0;

        if (!immersive) {
            if (elapsed > SESSION_DURATION + FADE_DURATION) {
                resetSession();
                elapsed = 0;
            }
            if (elapsed < FADE_DURATION) {
                var t = elapsed / FADE_DURATION;
                globalAlpha = t * t;
            }
            else if (elapsed > SESSION_DURATION) {
                var t = 1.0 - (elapsed - SESSION_DURATION) / FADE_DURATION;
                globalAlpha = t * t;
            }
        }

        var targetImm = immersive ? 1.0 : 0.0;
        immersiveT += (targetImm - immersiveT) * 0.05;

        rectTimer++;
        if (rectTimer > 60) {
            rectTimer = 0;
            measureContent();
        }

        if (!frameData || frameN === 0) {
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            return;
        }

        var palette = getPalette();
        var n = Math.min(frameN, maxVerts);
        var i, base, off, px, py, vm, sp, fmag, nrg, ageFrac, c;
        var alpha, ptSize;

        var normalAlpha = 0.08;
        var immAlpha = 0.35 + immersiveGlow * 0.55;
        var baseAlpha = normalAlpha + immersiveT * (immAlpha - normalAlpha);

        var normalPt = Math.max(1.5, Math.min(2.5, W / 600));
        var immPt = Math.max(3.0, Math.min(5.0, W / 350));
        var basePt = normalPt + immersiveT * (immPt - normalPt);

        for (i = 0; i < n; i++) {
            base = i * FLOATS_PER_PARTICLE;
            px = frameData[base];
            py = frameData[base + 1];
            vm = frameData[base + 2];
            sp = frameData[base + 3] | 0;
            fmag = frameData[base + 4];
            nrg = frameData[base + 5];
            ageFrac = frameData[base + 6];

            off = i * 8;
            vertexData[off] = px * dpr;
            vertexData[off + 1] = py * dpr;

            c = palette[sp % palette.length];
            var colorBoost = 1.0 + immersiveT * 0.5 + Math.min(fmag * 0.003, 0.15);
            var energyDim = lifecycle.enabled ? Math.max(nrg, 0.2) : 1.0;
            vertexData[off + 2] = Math.min(c[0] * colorBoost * energyDim, 1.0);
            vertexData[off + 3] = Math.min(c[1] * colorBoost * energyDim, 1.0);
            vertexData[off + 4] = Math.min(c[2] * colorBoost * energyDim, 1.0);

            var activity = Math.min(vm * 0.015, 0.08);
            alpha = (baseAlpha + activity * immersiveT) * globalAlpha;
            if (lifecycle.enabled) {
                alpha *= Math.min(nrg + 0.3, 1.0);
            }
            vertexData[off + 5] = alpha;

            ptSize = basePt + Math.min(vm * 0.03, 0.8) * immersiveT;
            if (lifecycle.enabled && ageFrac > 0.8) {
                ptSize *= 1.0 - (ageFrac - 0.8) * 2.0;
            }
            vertexData[off + 6] = ptSize * dpr * Math.max(globalAlpha, 0.05);
            vertexData[off + 7] = speciesShapes[sp % speciesShapes.length];
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertexData.subarray(0, n * 8));
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform2f(uResLoc, canvas.width, canvas.height);
        gl.drawArrays(gl.POINTS, 0, n);
    }

    document.addEventListener('visibilitychange', function () {
        if (!started) return;
        if (document.hidden) {
            running = false;
            if (animId) {
                cancelAnimationFrame(animId);
                animId = null;
            }
        }
        else if (immersive) {
            running = true;
            prevTime = performance.now();
            animId = requestAnimationFrame(render);
        }
    });

    function makeDraggable(el) {
        var handle = document.createElement('div');
        handle.className = 'pl-drag-handle';
        handle.textContent = '⠿';
        el.insertBefore(handle, el.firstChild);

        var startX, startY, origX, origY, dragging = false;

        function onDown(e) {
            dragging = true;
            var ev = e.touches ? e.touches[0] : e;
            startX = ev.clientX;
            startY = ev.clientY;
            var rect = el.getBoundingClientRect();
            origX = rect.left;
            origY = rect.top;
            e.preventDefault();
        }
        function onMove(e) {
            if (!dragging) {
                return;
            }
            var ev = e.touches ? e.touches[0] : e;
            var dx = ev.clientX - startX;
            var dy = ev.clientY - startY;
            el.style.position = 'fixed';
            el.style.left = (origX + dx) + 'px';
            el.style.top = (origY + dy) + 'px';
            el.style.right = 'auto';
            el.style.bottom = 'auto';
            el.style.transform = 'none';
        }
        function onUp() {
            dragging = false;
        }

        handle.addEventListener('mousedown', onDown);
        handle.addEventListener('touchstart', onDown, { passive: false });
        document.addEventListener('mousemove', onMove);
        document.addEventListener('touchmove', onMove, { passive: false });
        document.addEventListener('mouseup', onUp);
        document.addEventListener('touchend', onUp);
    }

    function buildControls() {
        if (controlsPanel) {
            return;
        }
        controlsPanel = document.createElement('div');
        controlsPanel.className = 'pl-controls';

        var sliders = [
            { label: 'Species', id: 'pl-species', min: 2, max: 12, val: NUM_SPECIES, step: 1 },
            { label: 'Count', id: 'pl-count', min: 2000, max: 30000, val: NUM_PARTICLES, step: 1000 },
            { label: 'Friction', id: 'pl-friction', min: 1, max: 30, val: Math.round(FRICTION * 100), step: 1 },
            { label: 'Force', id: 'pl-force', min: 1, max: 50, val: Math.round(FORCE_SCALE), step: 1 },
            { label: 'Range', id: 'pl-range', min: 20, max: 200, val: Math.round(RMAX), step: 5 },
            { label: 'Speed', id: 'pl-speed', min: 1, max: 20, val: Math.round(DT * 1000), step: 1 },
            { label: 'Repulsion', id: 'pl-repulsion', min: 10, max: 300, val: Math.round(REPULSION_SCALE * 100), step: 10 },
            { label: 'Damping', id: 'pl-damping', min: 50, max: 100, val: Math.round(VEL_DAMPING * 100), step: 1 },
            { label: 'Bounce', id: 'pl-bounce', min: 0, max: 100, val: 30, step: 5 },
            { label: 'Glow', id: 'pl-glow', min: 10, max: 100, val: Math.round(immersiveGlow * 100), step: 5 },
        ];

        var html = '<div class="pl-mode-toggle">' +
            '<button id="pl-mode-simple" class="pl-mode-btn' + (editorMode === 'simple' ? ' active' : '') + '">Simple</button>' +
            '<button id="pl-mode-advanced" class="pl-mode-btn' + (editorMode === 'advanced' ? ' active' : '') + '">Advanced</button>' +
            '</div>';
        html += '<div class="pl-controls-inner">';
        for (var i = 0; i < sliders.length; i++) {
            var s = sliders[i];
            html += '<div class="pl-ctrl-group">' +
                '<span class="pl-ctrl-label">' + s.label + '</span>' +
                '<input type="range" min="' + s.min + '" max="' + s.max + '" value="' + s.val + '" step="' + s.step + '" id="' + s.id + '">' +
                '<span class="pl-ctrl-val" id="' + s.id + '-val">' + s.val + '</span>' +
                '</div>';
        }
        html += '<div class="pl-ctrl-group">' +
            '<span class="pl-ctrl-label">Walls</span>' +
            '<select id="pl-walls" class="pl-select">' +
            '<option value="0"' + (WALL_MODE === 0 ? ' selected' : '') + '>Bounce</option>' +
            '<option value="1"' + (WALL_MODE === 1 ? ' selected' : '') + '>Wrap</option>' +
            '</select></div>';
        html += '<button id="pl-reseed">Reseed</button>';
        html += '<button id="pl-chaos">Chaos</button>';
        html += '<button id="pl-export" title="Download config">Export</button>';
        html += '<label class="pl-import-label" title="Load config"><input type="file" id="pl-import" accept=".json" hidden>Import</label>';
        html += '</div>';

        html += '<div class="pl-lifecycle-section">';
        html += '<div class="pl-lifecycle-header">' +
            '<label class="pl-lc-toggle"><input type="checkbox" id="pl-lc-enabled"' + (lifecycle.enabled ? ' checked' : '') + '> Lifecycle</label></div>';
        var lcSliders = [
            { label: 'Reproduction', id: 'pl-lc-repro', min: 0, max: 100, val: Math.round(lifecycle.reproRate * 100), step: 5 },
            { label: 'Mutation', id: 'pl-lc-mutation', min: 0, max: 100, val: Math.round(lifecycle.mutationRate * 100), step: 5 },
            { label: 'Lifespan', id: 'pl-lc-lifespan', min: 0, max: 10000, val: lifecycle.lifespan, step: 200 },
            { label: 'Energy+', id: 'pl-lc-egain', min: 0, max: 100, val: Math.round(lifecycle.energyGain * 100), step: 5 },
            { label: 'Energy-', id: 'pl-lc-ecost', min: 0, max: 100, val: Math.round(lifecycle.energyCost * 100), step: 5 },
            { label: 'Predation', id: 'pl-lc-predation', min: 0, max: 100, val: Math.round(lifecycle.predationStr * 100), step: 5 },
        ];
        html += '<div class="pl-lc-controls" id="pl-lc-controls">';
        for (var i = 0; i < lcSliders.length; i++) {
            var s = lcSliders[i];
            html += '<div class="pl-ctrl-group">' +
                '<span class="pl-ctrl-label">' + s.label + '</span>' +
                '<input type="range" min="' + s.min + '" max="' + s.max + '" value="' + s.val + '" step="' + s.step + '" id="' + s.id + '">' +
                '<span class="pl-ctrl-val" id="' + s.id + '-val">' + s.val + '</span>' +
                '</div>';
        }
        html += '</div></div>';
        html += '<div class="pl-hint">Esc: exit &middot; Drag ⠿ to move panels</div>';

        controlsPanel.innerHTML = html;
        document.body.appendChild(controlsPanel);
        makeDraggable(controlsPanel);

        function bind(id, fn) {
            document.getElementById(id).addEventListener('input', function (e) {
                var v = parseFloat(e.target.value);
                document.getElementById(id + '-val').textContent = Math.round(v);
                fn(v);
            });
        }

        bind('pl-species', function (v) {
            NUM_SPECIES = v | 0;
            resetSession();
        });
        bind('pl-count', function (v) {
            NUM_PARTICLES = v | 0;
            reallocVertexData();
            resetSession();
        });
        bind('pl-friction', function (v) {
            FRICTION = v / 100;
            sendUpdate('friction', FRICTION);
        });
        bind('pl-force', function (v) {
            FORCE_SCALE = v;
            sendUpdate('forceScale', FORCE_SCALE);
        });
        bind('pl-range', function (v) {
            RMAX = v;
            sendUpdate('rmax', RMAX);
        });
        bind('pl-speed', function (v) {
            DT = v / 1000;
            sendUpdate('dt', DT);
        });
        bind('pl-repulsion', function (v) {
            REPULSION_SCALE = v / 100;
            sendUpdate('repulsionScale', REPULSION_SCALE);
        });
        bind('pl-damping', function (v) {
            VEL_DAMPING = v / 100;
            sendUpdate('velDamping', VEL_DAMPING);
        });
        bind('pl-bounce', function (v) {
            sendUpdate('bounce', v / 100);
        });
        bind('pl-glow', function (v) {
            immersiveGlow = v / 100;
        });

        document.getElementById('pl-walls').addEventListener('change', function (e) {
            WALL_MODE = parseInt(e.target.value);
            sendUpdate('wallMode', WALL_MODE);
        });

        document.getElementById('pl-reseed').addEventListener('click', function () {
            resetSession();
        });
        document.getElementById('pl-chaos').addEventListener('click', function () {
            FORCE_SCALE = 10 + Math.random() * 35;
            FRICTION = 0.01 + Math.random() * 0.1;
            RMAX = 40 + Math.random() * 120;
            REPULSION_SCALE = 0.5 + Math.random() * 2.5;
            resetSession();
            syncSlider('pl-force', Math.round(FORCE_SCALE));
            syncSlider('pl-friction', Math.round(FRICTION * 100));
            syncSlider('pl-range', Math.round(RMAX));
            syncSlider('pl-repulsion', Math.round(REPULSION_SCALE * 100));
        });

        document.getElementById('pl-export').addEventListener('click', exportConfig);
        document.getElementById('pl-import').addEventListener('change', function (e) {
            if (e.target.files && e.target.files[0]) {
                importConfig(e.target.files[0]);
                e.target.value = '';
            }
        });

        document.getElementById('pl-mode-simple').addEventListener('click', function () {
            setEditorMode('simple');
        });
        document.getElementById('pl-mode-advanced').addEventListener('click', function () {
            setEditorMode('advanced');
        });

        document.getElementById('pl-lc-enabled').addEventListener('change', function (e) {
            lifecycle.enabled = e.target.checked;
            sendUpdate('lifecycle', lifecycle);
            var lcCtrl = document.getElementById('pl-lc-controls');
            if (lcCtrl) {
                lcCtrl.style.display = lifecycle.enabled ? 'flex' : 'none';
            }
        });
        var lcCtrl = document.getElementById('pl-lc-controls');
        if (lcCtrl) {
            lcCtrl.style.display = lifecycle.enabled ? 'flex' : 'none';
        }

        bind('pl-lc-repro', function (v) {
            lifecycle.reproRate = v / 100;
            sendUpdate('lifecycle', lifecycle);
        });
        bind('pl-lc-mutation', function (v) {
            lifecycle.mutationRate = v / 100;
            sendUpdate('lifecycle', lifecycle);
        });
        bind('pl-lc-lifespan', function (v) {
            lifecycle.lifespan = v;
            sendUpdate('lifecycle', lifecycle);
        });
        bind('pl-lc-egain', function (v) {
            lifecycle.energyGain = v / 100;
            sendUpdate('lifecycle', lifecycle);
        });
        bind('pl-lc-ecost', function (v) {
            lifecycle.energyCost = v / 100;
            sendUpdate('lifecycle', lifecycle);
        });
        bind('pl-lc-predation', function (v) {
            lifecycle.predationStr = v / 100;
            sendUpdate('lifecycle', lifecycle);
        });
    }

    function syncSlider(id, val) {
        var el = document.getElementById(id);
        if (el) {
            el.value = val;
            document.getElementById(id + '-val').textContent = val;
        }
    }

    function setEditorMode(mode) {
        editorMode = mode;
        var sBtn = document.getElementById('pl-mode-simple');
        var aBtn = document.getElementById('pl-mode-advanced');
        if (sBtn && aBtn) {
            sBtn.classList.toggle('active', mode === 'simple');
            aBtn.classList.toggle('active', mode === 'advanced');
        }
        if (mode === 'advanced') {
            buildSpeciesPanel();
        }
        else {
            removeSpeciesPanel();
        }
    }

    function buildSpeciesPanel() {
        if (speciesPanel) {
            speciesPanel.remove();
        }
        speciesPanel = document.createElement('div');
        speciesPanel.className = 'pl-species-panel';
        rebuildSpeciesPanel();
        document.body.appendChild(speciesPanel);
        makeDraggable(speciesPanel);
    }

    function rebuildSpeciesPanel() {
        if (!speciesPanel || !currentAttractions) {
            return;
        }
        var n = NUM_SPECIES;
        var palette = getPalette();

        var existingHandle = speciesPanel.querySelector('.pl-drag-handle');

        var html = '<div class="pl-sp-header">Species Editor</div>';
        html += '<div class="pl-sp-scroll">';

        html += '<div class="pl-sp-section">Attraction Matrix</div>';
        html += '<div class="pl-sp-note">Row \u2192 Column. Positive = attract, Negative = repel.</div>';
        html += '<table class="pl-sp-matrix"><thead><tr><th></th>';
        for (var j = 0; j < n; j++) {
            html += '<th><span class="pl-sp-dot" style="background:' + paletteToHex(palette[j]) + '"></span>' + speciesNames[j] + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (var i = 0; i < n; i++) {
            html += '<tr><td><span class="pl-sp-dot" style="background:' + paletteToHex(palette[i]) + '"></span>' + speciesNames[i] + '</td>';
            for (var j = 0; j < n; j++) {
                var val = Math.round(currentAttractions[i * n + j] * 100);
                html += '<td><input type="range" min="-100" max="100" value="' + val + '" class="pl-sp-slider" data-i="' + i + '" data-j="' + j + '" data-field="attraction"></td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        html += '<div class="pl-sp-section">Min Distance Matrix</div>';
        html += '<table class="pl-sp-matrix"><thead><tr><th></th>';
        for (var j = 0; j < n; j++) {
            html += '<th>' + speciesNames[j] + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (var i = 0; i < n; i++) {
            html += '<tr><td>' + speciesNames[i] + '</td>';
            for (var j = 0; j < n; j++) {
                var val = Math.round(currentMinDist[i * n + j] * 100);
                html += '<td><input type="range" min="5" max="60" value="' + val + '" class="pl-sp-slider" data-i="' + i + '" data-j="' + j + '" data-field="minDist"></td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        html += '</div>';

        if (existingHandle) {
            var wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            while (speciesPanel.childNodes.length > 1) {
                speciesPanel.removeChild(speciesPanel.lastChild);
            }
            while (wrapper.firstChild) {
                speciesPanel.appendChild(wrapper.firstChild);
            }
        }
        else {
            speciesPanel.innerHTML = html;
        }

        speciesPanel.addEventListener('input', function (e) {
            if (!e.target.classList.contains('pl-sp-slider')) {
                return;
            }
            var si = parseInt(e.target.dataset.i);
            var sj = parseInt(e.target.dataset.j);
            var field = e.target.dataset.field;
            var v = parseFloat(e.target.value);
            var idx = si * NUM_SPECIES + sj;

            if (field === 'attraction') {
                currentAttractions[idx] = v / 100;
                sendUpdate('attractions', currentAttractions);
            }
            else if (field === 'minDist') {
                currentMinDist[idx] = v / 100;
                sendUpdate('minDist', currentMinDist);
            }
        });
    }

    function removeSpeciesPanel() {
        if (speciesPanel) {
            speciesPanel.remove();
            speciesPanel = null;
        }
    }

    function removeControls() {
        if (controlsPanel) {
            controlsPanel.remove();
            controlsPanel = null;
        }
    }

    function startSimulation() {
        if (started) return;
        started = true;
        running = true;
        measureContent();
        initWorker();
        animId = requestAnimationFrame(render);
    }

    function enterImmersive() {
        immersive = true;
        startSimulation();
        document.body.classList.add('immersive-mode');
        canvas.style.pointerEvents = 'auto';
        buildControls();
        if (editorMode === 'advanced') {
            buildSpeciesPanel();
        }
        measureContent();
        if (worker) {
            worker.postMessage({
                type: 'update',
                zones: { left: W * 0.5, right: W * 0.5 }
            });
        }
    }

    function exitImmersive() {
        immersive = false;
        running = false;
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        document.body.classList.remove('immersive-mode');
        canvas.style.pointerEvents = 'none';
        removeControls();
        removeSpeciesPanel();
        if (worker) {
            worker.terminate();
            worker = null;
        }
        started = false;
        frameData = null;
        frameN = 0;
    }

    var btn = document.getElementById('immersive-toggle');
    if (btn) {
        btn.innerHTML = plIconSvg;
        btn.addEventListener('click', function () {
            if (immersive) {
                exitImmersive();
            }
            else {
                enterImmersive();
            }
        });

        setInterval(function () {
            if (!immersive && btn) {
                btn.classList.add('pl-btn-glow');
                setTimeout(function () {
                    btn.classList.remove('pl-btn-glow');
                }, 2000);
            }
        }, 30000);
    }

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && immersive) {
            exitImmersive();
        }
    });
})();
