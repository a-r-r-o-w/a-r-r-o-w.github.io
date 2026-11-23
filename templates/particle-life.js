(function () {
    var canvas = document.getElementById('particle-life');
    if (!canvas) return;

    var gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false, antialias: false });
    if (!gl) gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false });
    if (!gl) return;
    var isWebGL2 = !!gl.createTransformFeedback;

    var NUM_SPECIES = 6, NUM_PARTICLES = 5000;
    var RMAX = 85.0, FRICTION = 0.28, DT = 0.01, FORCE_SCALE = 7.0;
    var REPULSION_SCALE = 1.6, BOUNCE = 0.3, WALL_MODE = 1;
    var immersive = false;
    var W, H, dpr;
    var seed = (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0;
    var rngState = seed;
    var worker = null, running = false, animId = null;
    var displayBuf = null, displayView = null, displayN = 0, recycleBuf = null;
    var frameDirty = false, stepInFlight = false;
    var controlsPanel = null, speciesPanel = null;
    var editorMode = 'simple';
    var activePreset = null;
    var currentAttractions = null, currentMinDist = null;

    var FLOATS_PER_PARTICLE = 4;
    var STRIDE_BYTES = FLOATS_PER_PARTICLE * 4;

    var lifecycle = {
        enabled: false, reproRate: 0.5, mutationRate: 0.2,
        lifespan: 3000, energyGain: 0.5, energyCost: 0.3, predationStr: 0.5
    };

    var darkPalette = [
        [1.00, 0.22, 0.30], [0.10, 0.85, 1.00], [0.25, 1.00, 0.35],
        [1.00, 0.88, 0.10], [0.88, 0.28, 1.00], [1.00, 0.52, 0.05],
        [0.05, 1.00, 0.80], [1.00, 0.30, 0.78], [0.40, 0.68, 1.00],
        [0.80, 1.00, 0.15], [1.00, 0.62, 0.45], [0.58, 0.38, 1.00],
    ];
    var lightPalette = [
        [0.92, 0.12, 0.20], [0.00, 0.52, 0.90], [0.06, 0.68, 0.18],
        [0.86, 0.58, 0.00], [0.68, 0.12, 0.88], [0.94, 0.38, 0.00],
        [0.00, 0.66, 0.56], [0.88, 0.12, 0.55], [0.18, 0.40, 0.90],
        [0.48, 0.68, 0.00], [0.88, 0.36, 0.20], [0.40, 0.16, 0.88],
    ];
    var darkPaletteFlat = new Float32Array(12 * 3);
    var lightPaletteFlat = new Float32Array(12 * 3);
    for (var pi = 0; pi < 12; pi++) {
        darkPaletteFlat[pi * 3] = darkPalette[pi][0];
        darkPaletteFlat[pi * 3 + 1] = darkPalette[pi][1];
        darkPaletteFlat[pi * 3 + 2] = darkPalette[pi][2];
        lightPaletteFlat[pi * 3] = lightPalette[pi][0];
        lightPaletteFlat[pi * 3 + 1] = lightPalette[pi][1];
        lightPaletteFlat[pi * 3 + 2] = lightPalette[pi][2];
    }

    var speciesNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'];

    function fillMat(n, fn) {
        var out = new Array(n * n);
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) out[i * n + j] = fn(i, j, n);
        }
        return out;
    }

    var PRESETS = [
        {
            name: 'Snakes',
            numSpecies: 6, numParticles: 5000, rmax: 75, friction: 0.30, dt: 0.01,
            forceScale: 7, repulsionScale: 1.7, bounce: 0.3, wallMode: 1,
            attractions: fillMat(6, function (i, j, n) {
                var d = (j - i + n) % n;
                if (d === 0) return 0.85;
                if (d === 1) return 0.38;
                if (d === n - 1) return 0.14;
                return -0.75;
            }),
            minDist: fillMat(6, function (i, j) { return i === j ? 0.24 : 0.34; }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Living Cells',
            numSpecies: 4, numParticles: 5000, rmax: 90, friction: 0.32, dt: 0.01,
            forceScale: 6, repulsionScale: 1.7, bounce: 0.3, wallMode: 1,
            attractions: [
                0.90, 0.30, -0.80, 0.05,
                0.70, 0.45, -0.45, 0.20,
                -0.65, -0.35, -0.10, 0.05,
                0.05, 0.35, -0.15, 0.10
            ],
            minDist: [
                0.24, 0.34, 0.44, 0.28,
                0.34, 0.26, 0.38, 0.24,
                0.44, 0.38, 0.32, 0.26,
                0.28, 0.24, 0.26, 0.24
            ],
            lifecycle: { enabled: false }
        },
        {
            name: 'Gliders',
            numSpecies: 3, numParticles: 4500, rmax: 58, friction: 0.28, dt: 0.01,
            forceScale: 6, repulsionScale: 1.6, bounce: 0.3, wallMode: 1,
            attractions: [
                0.75, 0.85, -0.60,
                -0.60, 0.75, 0.85,
                0.85, -0.60, 0.75
            ],
            minDist: fillMat(3, function () { return 0.30; }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Crystals',
            numSpecies: 4, numParticles: 4000, rmax: 80, friction: 0.42, dt: 0.008,
            forceScale: 6, repulsionScale: 2.5, bounce: 0.3, wallMode: 0,
            attractions: fillMat(4, function (i, j) {
                if (i === j) return 0.35;
                return ((i + j) % 2 === 0) ? 0.25 : -0.15;
            }),
            minDist: fillMat(4, function () { return 0.56; }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Solar Systems',
            numSpecies: 5, numParticles: 5000, rmax: 95, friction: 0.30, dt: 0.01,
            forceScale: 5, repulsionScale: 1.8, bounce: 0.3, wallMode: 1,
            attractions: [
                0.70, -0.05, -0.05, -0.05, -0.05,
                0.65, 0.15, 0.30, -0.15, -0.25,
                0.50, -0.25, 0.15, 0.30, -0.15,
                0.35, -0.15, -0.25, 0.15, 0.30,
                0.25, 0.30, -0.15, -0.25, 0.15
            ],
            minDist: fillMat(5, function (i, j) {
                if (i === 0 && j === 0) return 0.24;
                if (i === 0 || j === 0) return 0.22 + Math.max(i, j) * 0.11;
                return 0.28;
            }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Predator & Prey',
            numSpecies: 4, numParticles: 5000, rmax: 75, friction: 0.28, dt: 0.01,
            forceScale: 6, repulsionScale: 1.6, bounce: 0.3, wallMode: 1,
            attractions: [
                0.25, 0.90, -0.20, -0.20,
                -0.80, 0.45, 0.85, -0.10,
                -0.10, -0.45, 0.65, 0.20,
                0.50, -0.25, 0.40, 0.35
            ],
            minDist: fillMat(4, function () { return 0.28; }),
            lifecycle: {
                enabled: true, reproRate: 0.55, mutationRate: 0.08,
                lifespan: 3600, energyGain: 0.45, energyCost: 0.25, predationStr: 0.70
            }
        },
        {
            name: 'Symbiosis',
            numSpecies: 6, numParticles: 5000, rmax: 80, friction: 0.30, dt: 0.01,
            forceScale: 7, repulsionScale: 1.7, bounce: 0.3, wallMode: 1,
            attractions: fillMat(6, function (i, j) {
                if (i === j) return 0.30;
                if ((i >> 1) === (j >> 1)) return 0.95;
                return -0.45;
            }),
            minDist: fillMat(6, function (i, j) {
                return (i >> 1) === (j >> 1) ? 0.24 : 0.38;
            }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Vortex',
            numSpecies: 6, numParticles: 5000, rmax: 95, friction: 0.26, dt: 0.01,
            forceScale: 5, repulsionScale: 1.6, bounce: 0.3, wallMode: 1,
            attractions: fillMat(6, function (i, j, n) {
                var d = (j - i + n) % n;
                if (d === 0) return 0.40;
                if (d === 1) return 0.75;
                if (d === 2) return 0.30;
                if (d === n - 1) return -0.50;
                if (d === n - 2) return -0.20;
                return 0.0;
            }),
            minDist: fillMat(6, function () { return 0.26; }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Mitosis',
            numSpecies: 4, numParticles: 4500, rmax: 85, friction: 0.34, dt: 0.01,
            forceScale: 6, repulsionScale: 1.9, bounce: 0.3, wallMode: 1,
            attractions: [
                0.80, 0.50, -0.60, -0.35,
                0.40, 0.75, 0.35, -0.60,
                -0.55, 0.25, 0.80, 0.45,
                -0.30, -0.55, 0.30, 0.75
            ],
            minDist: fillMat(4, function (i, j) { return i === j ? 0.34 : 0.28; }),
            lifecycle: {
                enabled: true, reproRate: 0.60, mutationRate: 0.10,
                lifespan: 3000, energyGain: 0.55, energyCost: 0.26, predationStr: 0.35
            }
        },
        {
            name: 'Membranes',
            numSpecies: 3, numParticles: 5000, rmax: 90, friction: 0.34, dt: 0.01,
            forceScale: 6, repulsionScale: 1.8, bounce: 0.3, wallMode: 1,
            attractions: [
                0.65, -0.80, 0.45,
                -0.80, -0.10, -0.65,
                0.45, -0.65, 0.55
            ],
            minDist: [
                0.26, 0.44, 0.30,
                0.44, 0.36, 0.42,
                0.30, 0.42, 0.26
            ],
            lifecycle: { enabled: false }
        },
        {
            name: 'Swarm',
            numSpecies: 4, numParticles: 5000, rmax: 75, friction: 0.28, dt: 0.01,
            forceScale: 6, repulsionScale: 1.6, bounce: 0.3, wallMode: 1,
            attractions: fillMat(4, function (i, j, n) {
                var d = (j - i + n) % n;
                if (d === 0) return 0.55;
                if (d === 1) return 0.65;
                if (d === n - 1) return -0.35;
                return -0.10;
            }),
            minDist: fillMat(4, function () { return 0.28; }),
            lifecycle: { enabled: false }
        },
        {
            name: 'Primordial Soup',
            numSpecies: 8, numParticles: 5500, rmax: 80, friction: 0.28, dt: 0.01,
            forceScale: 6, repulsionScale: 1.6, bounce: 0.3, wallMode: 1,
            attractions: fillMat(8, function (i, j, n) {
                var d = (j - i + n) % n;
                if (d === 0) return 0.65;
                if (d === 1) return 0.75;
                if (d === 2) return 0.20;
                if (d === n - 1) return -0.55;
                if (d === n - 2) return -0.20;
                return ((i + j) % 2 === 0 ? 0.15 : -0.25);
            }),
            minDist: fillMat(8, function () { return 0.26; }),
            lifecycle: {
                enabled: true, reproRate: 0.50, mutationRate: 0.35,
                lifespan: 3200, energyGain: 0.48, energyCost: 0.26, predationStr: 0.55
            }
        }
    ];

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

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        W = window.innerWidth;
        H = window.innerHeight;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        gl.viewport(0, 0, canvas.width, canvas.height);
        if (worker) {
            worker.postMessage({ type: 'update', resize: { worldW: W, worldH: H } });
        }
    }

    function generateRules() {
        var n = NUM_SPECIES;
        var a = new Float32Array(n * n);
        var m = new Float32Array(n * n);
        for (var i = 0; i < n * n; i++) {
            a[i] = (xorshift32() * 2.0 - 1.0) * 0.8;
            m[i] = 0.15 + xorshift32() * 0.25;
        }
        return { attractions: a, minDist: m };
    }

    function startWorker(keepRules) {
        if (worker) worker.terminate();
        if (!keepRules || !currentAttractions || !currentMinDist) {
            var rules = generateRules();
            currentAttractions = rules.attractions;
            currentMinDist = rules.minDist;
        }

        displayBuf = null;
        displayView = null;
        displayN = 0;
        recycleBuf = new ArrayBuffer(Math.max(NUM_PARTICLES, 40000) * STRIDE_BYTES);
        frameDirty = false;
        stepInFlight = true;

        worker = new Worker('/assets/particle-worker.js');
        worker.onmessage = function (e) {
            var d = e.data;
            if (d.type === 'frame') {
                if (displayBuf) recycleBuf = displayBuf;
                displayBuf = d.buf;
                displayView = new Float32Array(displayBuf, 0, d.n * FLOATS_PER_PARTICLE);
                displayN = d.n;
                frameDirty = true;
                stepInFlight = false;
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
                seed: seed, repulsionScale: REPULSION_SCALE,
                bounce: BOUNCE, wallMode: WALL_MODE,
                lifecycle: lifecycle,
            }
        });

        if (speciesPanel && editorMode === 'advanced') {
            rebuildSpeciesPanel();
        }
    }

    function resetSession(keepRules) {
        seed = (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0;
        rngState = seed;
        startWorker(keepRules);
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
            version: 3, numSpecies: n, numParticles: NUM_PARTICLES,
            rmax: RMAX, friction: FRICTION, dt: DT, forceScale: FORCE_SCALE,
            repulsionScale: REPULSION_SCALE, bounce: BOUNCE,
            wallMode: WALL_MODE,
            attractions: att, minDist: md, lifecycle: lifecycle,
        };
        var blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'particle-life-config.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function applyConfig(cfg, presetName) {
        if (!cfg.attractions || !cfg.minDist) return;
        activePreset = presetName || null;
        NUM_SPECIES = cfg.numSpecies || 6;
        NUM_PARTICLES = cfg.numParticles || 5000;
        RMAX = cfg.rmax || 85;
        FRICTION = cfg.friction || 0.28;
        DT = cfg.dt || 0.01;
        FORCE_SCALE = cfg.forceScale || 7;
        REPULSION_SCALE = cfg.repulsionScale || 1.6;
        BOUNCE = cfg.bounce !== undefined ? cfg.bounce : 0.3;
        WALL_MODE = cfg.wallMode !== undefined ? cfg.wallMode : 1;
        lifecycle = Object.assign({
            enabled: false, reproRate: 0.5, mutationRate: 0.2,
            lifespan: 3000, energyGain: 0.5, energyCost: 0.3, predationStr: 0.5
        }, cfg.lifecycle || { enabled: false });
        currentAttractions = new Float32Array(cfg.attractions);
        currentMinDist = new Float32Array(cfg.minDist);
        reallocVbo();
        resetSession(true);
        var pos = controlsPanel ? {
            left: controlsPanel.style.left, top: controlsPanel.style.top,
            right: controlsPanel.style.right, bottom: controlsPanel.style.bottom,
            transform: controlsPanel.style.transform
        } : null;
        removeControls();
        removeSpeciesPanel();
        buildControls();
        if (pos && pos.left) {
            controlsPanel.style.left = pos.left;
            controlsPanel.style.top = pos.top;
            controlsPanel.style.right = pos.right;
            controlsPanel.style.bottom = pos.bottom;
            controlsPanel.style.transform = pos.transform;
        }
        if (editorMode === 'advanced') buildSpeciesPanel();
    }

    function importConfig(file) {
        var reader = new FileReader();
        reader.onload = function (e) {
            try { applyConfig(JSON.parse(e.target.result), null); }
            catch (err) { }
        };
        reader.readAsText(file);
    }

    var vsSrc = isWebGL2
        ? '#version 300 es\nin vec2 aPos;\nin vec2 aMeta;\nuniform vec2 uRes;\nuniform vec3 uPalette[12];\nuniform float uPointSize;\nuniform float uBaseAlpha;\nuniform float uColorBoost;\nout vec4 vColor;\nout float vShape;\nvoid main() {\n    vec2 c = (aPos / uRes) * 2.0 - 1.0;\n    c.y = -c.y;\n    gl_Position = vec4(c, 0.0, 1.0);\n    int sp = int(aMeta.x + 0.5);\n    float dim = aMeta.y;\n    gl_PointSize = uPointSize * (0.65 + 0.35 * dim);\n    vColor = vec4(min(uPalette[sp] * uColorBoost * dim, 1.0), min(uBaseAlpha * dim, 1.0));\n    vShape = mod(aMeta.x, 6.0);\n}\n'
        : 'attribute vec2 aPos;\nattribute vec2 aMeta;\nuniform vec2 uRes;\nuniform vec3 uPalette[12];\nuniform float uPointSize;\nuniform float uBaseAlpha;\nuniform float uColorBoost;\nvarying vec4 vColor;\nvarying float vShape;\nvoid main() {\n    vec2 c = (aPos / uRes) * 2.0 - 1.0;\n    c.y = -c.y;\n    gl_Position = vec4(c, 0.0, 1.0);\n    int sp = int(aMeta.x + 0.5);\n    float dim = aMeta.y;\n    gl_PointSize = uPointSize * (0.65 + 0.35 * dim);\n    vColor = vec4(min(uPalette[sp] * uColorBoost * dim, 1.0), min(uBaseAlpha * dim, 1.0));\n    vShape = mod(aMeta.x, 6.0);\n}\n';

    var fsSrc = isWebGL2
        ? '#version 300 es\nprecision mediump float;\nin vec4 vColor;\nin float vShape;\nout vec4 fc;\nvoid main() {\n    vec2 d = gl_PointCoord - 0.5;\n    float r2 = dot(d, d);\n    int s = int(vShape + 0.5);\n    float mask = 0.0;\n    if (s == 0) {\n        mask = smoothstep(0.25, 0.10, r2);\n    } else if (s == 1) {\n        float manh = abs(d.x) + abs(d.y);\n        mask = smoothstep(0.42, 0.34, manh);\n    } else if (s == 2) {\n        float box = max(abs(d.x), abs(d.y));\n        mask = smoothstep(0.38, 0.30, box);\n    } else if (s == 3) {\n        float tx = d.x * 1.8, ty = d.y + 0.15;\n        float tri = max(abs(tx) - 0.5 + ty, -ty - 0.30);\n        mask = smoothstep(0.02, -0.02, tri);\n    } else if (s == 4) {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float star = r - 0.25 - 0.12 * cos(a * 5.0);\n        mask = smoothstep(0.02, -0.02, star);\n    } else {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float hex = r - 0.35 + 0.06 * cos(a * 6.0);\n        mask = smoothstep(0.02, -0.02, hex);\n    }\n    if (mask < 0.01) discard;\n    fc = vec4(vColor.rgb, vColor.a * mask);\n}\n'
        : 'precision mediump float;\nvarying vec4 vColor;\nvarying float vShape;\nvoid main() {\n    vec2 d = gl_PointCoord - 0.5;\n    float r2 = dot(d, d);\n    int s = int(vShape + 0.5);\n    float mask = 0.0;\n    if (s == 0) {\n        mask = smoothstep(0.25, 0.10, r2);\n    } else if (s == 1) {\n        float manh = abs(d.x) + abs(d.y);\n        mask = smoothstep(0.42, 0.34, manh);\n    } else if (s == 2) {\n        float box = max(abs(d.x), abs(d.y));\n        mask = smoothstep(0.38, 0.30, box);\n    } else if (s == 3) {\n        float tx = d.x * 1.8, ty = d.y + 0.15;\n        float tri = max(abs(tx) - 0.5 + ty, -ty - 0.30);\n        mask = smoothstep(0.02, -0.02, tri);\n    } else if (s == 4) {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float star = r - 0.25 - 0.12 * cos(a * 5.0);\n        mask = smoothstep(0.02, -0.02, star);\n    } else {\n        float r = sqrt(r2);\n        float a = atan(d.y, d.x);\n        float hex = r - 0.35 + 0.06 * cos(a * 6.0);\n        mask = smoothstep(0.02, -0.02, hex);\n    }\n    if (mask < 0.01) discard;\n    gl_FragColor = vec4(vColor.rgb, vColor.a * mask);\n}\n';

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
    if (!vs || !fs) return;
    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    var aPosLoc = gl.getAttribLocation(prog, 'aPos');
    var aMetaLoc = gl.getAttribLocation(prog, 'aMeta');
    var uResLoc = gl.getUniformLocation(prog, 'uRes');
    var uPaletteLoc = gl.getUniformLocation(prog, 'uPalette');
    var uPointSizeLoc = gl.getUniformLocation(prog, 'uPointSize');
    var uBaseAlphaLoc = gl.getUniformLocation(prog, 'uBaseAlpha');
    var uColorBoostLoc = gl.getUniformLocation(prog, 'uColorBoost');

    var maxVerts = 40000;
    var vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, maxVerts * STRIDE_BYTES, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, STRIDE_BYTES, 0);
    gl.enableVertexAttribArray(aMetaLoc);
    gl.vertexAttribPointer(aMetaLoc, 2, gl.FLOAT, false, STRIDE_BYTES, 8);
    gl.enable(gl.BLEND);

    function reallocVbo() {
        var needed = Math.max(NUM_PARTICLES, maxVerts);
        if (needed > maxVerts) {
            maxVerts = needed;
            gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
            gl.bufferData(gl.ARRAY_BUFFER, maxVerts * STRIDE_BYTES, gl.DYNAMIC_DRAW);
        }
    }

    resize();
    window.addEventListener('resize', resize);

    function isLight() {
        return document.documentElement.dataset.theme === 'light';
    }

    function getPalette() {
        return isLight() ? lightPalette : darkPalette;
    }

    function render() {
        if (!running) return;
        animId = requestAnimationFrame(render);

        if (!stepInFlight && recycleBuf && worker) {
            var b = recycleBuf;
            recycleBuf = null;
            stepInFlight = true;
            worker.postMessage({ type: 'step', buf: b }, [b]);
        }

        if (!frameDirty || !displayView || displayN === 0) return;
        frameDirty = false;

        var light = isLight();
        gl.blendFunc(gl.SRC_ALPHA, light ? gl.ONE_MINUS_SRC_ALPHA : gl.ONE);
        gl.uniform2f(uResLoc, W, H);
        gl.uniform3fv(uPaletteLoc, light ? lightPaletteFlat : darkPaletteFlat);
        gl.uniform1f(uPointSizeLoc, Math.max(3.8, Math.min(6.0, W / 290)) * dpr);
        gl.uniform1f(uBaseAlphaLoc, 0.92);
        gl.uniform1f(uColorBoostLoc, light ? 1.05 : 1.65);

        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, displayView);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.POINTS, 0, displayN);
    }

    var pointerMode = 0;
    function sendMouse(x, y, mode) {
        if (worker) worker.postMessage({ type: 'mouse', x: x, y: y, mode: mode });
    }
    canvas.addEventListener('contextmenu', function (e) {
        if (immersive) e.preventDefault();
    });
    canvas.addEventListener('mousedown', function (e) {
        if (!immersive) return;
        pointerMode = e.button === 2 ? -1 : 1;
        sendMouse(e.clientX, e.clientY, pointerMode);
    });
    canvas.addEventListener('mousemove', function (e) {
        if (!immersive || pointerMode === 0) return;
        sendMouse(e.clientX, e.clientY, pointerMode);
    });
    window.addEventListener('mouseup', function () {
        if (pointerMode !== 0) {
            pointerMode = 0;
            sendMouse(0, 0, 0);
        }
    });
    canvas.addEventListener('touchstart', function (e) {
        if (!immersive || e.touches.length !== 1) return;
        pointerMode = 1;
        sendMouse(e.touches[0].clientX, e.touches[0].clientY, 1);
    }, { passive: true });
    canvas.addEventListener('touchmove', function (e) {
        if (!immersive || pointerMode === 0 || e.touches.length !== 1) return;
        sendMouse(e.touches[0].clientX, e.touches[0].clientY, 1);
    }, { passive: true });
    window.addEventListener('touchend', function () {
        if (pointerMode !== 0) {
            pointerMode = 0;
            sendMouse(0, 0, 0);
        }
    });

    document.addEventListener('visibilitychange', function () {
        if (!immersive) return;
        if (document.hidden) {
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
        } else {
            running = true;
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
            if (!dragging) return;
            var ev = e.touches ? e.touches[0] : e;
            el.style.position = 'fixed';
            el.style.left = (origX + ev.clientX - startX) + 'px';
            el.style.top = (origY + ev.clientY - startY) + 'px';
            el.style.right = 'auto';
            el.style.bottom = 'auto';
            el.style.transform = 'none';
        }
        function onUp() { dragging = false; }

        handle.addEventListener('mousedown', onDown);
        handle.addEventListener('touchstart', onDown, { passive: false });
        document.addEventListener('mousemove', onMove);
        document.addEventListener('touchmove', onMove, { passive: false });
        document.addEventListener('mouseup', onUp);
        document.addEventListener('touchend', onUp);

        el._plCleanup = function () {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('touchmove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.removeEventListener('touchend', onUp);
        };
    }

    function cleanupPanel(el) {
        if (!el) return;
        if (el._plCleanup) el._plCleanup();
        el.remove();
    }

    function buildControls() {
        if (controlsPanel) return;
        controlsPanel = document.createElement('div');
        controlsPanel.className = 'pl-controls';

        var sliders = [
            { label: 'Species', id: 'pl-species', min: 2, max: 12, val: NUM_SPECIES, step: 1 },
            { label: 'Count', id: 'pl-count', min: 1000, max: 20000, val: NUM_PARTICLES, step: 1000 },
            { label: 'Friction', id: 'pl-friction', min: 1, max: 60, val: Math.round(FRICTION * 100), step: 1 },
            { label: 'Force', id: 'pl-force', min: 1, max: 30, val: Math.round(FORCE_SCALE), step: 1 },
            { label: 'Range', id: 'pl-range', min: 20, max: 200, val: Math.round(RMAX), step: 5 },
            { label: 'Speed', id: 'pl-speed', min: 1, max: 25, val: Math.round(DT * 1000), step: 1 },
            { label: 'Repulsion', id: 'pl-repulsion', min: 10, max: 300, val: Math.round(REPULSION_SCALE * 100), step: 10 },
            { label: 'Bounce', id: 'pl-bounce', min: 0, max: 100, val: Math.round(BOUNCE * 100), step: 5 },
        ];

        var html = '<div class="pl-mode-toggle">' +
            '<button id="pl-mode-simple" class="pl-mode-btn' + (editorMode === 'simple' ? ' active' : '') + '">Simple</button>' +
            '<button id="pl-mode-advanced" class="pl-mode-btn' + (editorMode === 'advanced' ? ' active' : '') + '">Advanced</button>' +
            '<button id="pl-mode-presets" class="pl-mode-btn' + (editorMode === 'presets' ? ' active' : '') + '">Presets</button>' +
            '<button id="pl-share" class="pl-mode-btn" title="Copy shareable link">Share</button>' +
            '</div>';

        var showKnobs = editorMode !== 'presets';
        html += '<div id="pl-knobs-view" style="display:' + (showKnobs ? 'block' : 'none') + '">';
        html += '<div class="pl-controls-inner">';
        for (var i = 0; i < sliders.length; i++) {
            var s = sliders[i];
            html += '<div class="pl-ctrl-group">' +
                '<span class="pl-ctrl-label">' + s.label + '</span>' +
                '<input type="range" min="' + s.min + '" max="' + s.max + '" value="' + s.val + '" step="' + s.step + '" id="' + s.id + '">' +
                '<span class="pl-ctrl-val" id="' + s.id + '-val">' + s.val + '</span>' +
                '</div>';
        }
        html += '</div>';

        html += '<div class="pl-actions-row">' +
            '<div class="pl-ctrl-group">' +
            '<span class="pl-ctrl-label">Walls</span>' +
            '<select id="pl-walls" class="pl-select">' +
            '<option value="0"' + (WALL_MODE === 0 ? ' selected' : '') + '>Bounce</option>' +
            '<option value="1"' + (WALL_MODE === 1 ? ' selected' : '') + '>Wrap</option>' +
            '</select></div>' +
            '<button id="pl-reseed">Reseed</button>' +
            '<button id="pl-chaos">Chaos</button>' +
            '<button id="pl-export" title="Download config">Export</button>' +
            '<label class="pl-import-label" title="Load config"><input type="file" id="pl-import" accept=".json" hidden>Import</label>' +
            '</div>';

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
        for (var j = 0; j < lcSliders.length; j++) {
            var ls = lcSliders[j];
            html += '<div class="pl-ctrl-group">' +
                '<span class="pl-ctrl-label">' + ls.label + '</span>' +
                '<input type="range" min="' + ls.min + '" max="' + ls.max + '" value="' + ls.val + '" step="' + ls.step + '" id="' + ls.id + '">' +
                '<span class="pl-ctrl-val" id="' + ls.id + '-val">' + ls.val + '</span>' +
                '</div>';
        }
        html += '</div></div></div>';

        html += '<div id="pl-presets-view" class="pl-presets-grid" style="display:' + (showKnobs ? 'none' : 'grid') + '">';
        for (var p = 0; p < PRESETS.length; p++) {
            var pr = PRESETS[p];
            var isAct = activePreset === pr.name ? ' active' : '';
            html += '<button class="pl-preset-btn' + isAct + '" data-preset="' + p + '">' + pr.name + '</button>';
        }
        html += '</div>';

        html += '<div class="pl-hint">Left-drag: attract &middot; Right-drag: repel &middot; Esc: exit &middot; Drag ⠿ to move</div>';

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

        bind('pl-species', function (v) { activePreset = null; NUM_SPECIES = v | 0; resetSession(false); });
        bind('pl-count', function (v) { NUM_PARTICLES = v | 0; reallocVbo(); resetSession(true); });
        bind('pl-friction', function (v) { FRICTION = v / 100; sendUpdate('friction', FRICTION); });
        bind('pl-force', function (v) { FORCE_SCALE = v; sendUpdate('forceScale', FORCE_SCALE); });
        bind('pl-range', function (v) { RMAX = v; sendUpdate('rmax', RMAX); });
        bind('pl-speed', function (v) { DT = v / 1000; sendUpdate('dt', DT); });
        bind('pl-repulsion', function (v) { REPULSION_SCALE = v / 100; sendUpdate('repulsionScale', REPULSION_SCALE); });
        bind('pl-bounce', function (v) { BOUNCE = v / 100; sendUpdate('bounce', BOUNCE); });

        document.getElementById('pl-walls').addEventListener('change', function (e) {
            WALL_MODE = parseInt(e.target.value, 10);
            sendUpdate('wallMode', WALL_MODE);
        });

        document.getElementById('pl-reseed').addEventListener('click', function () { activePreset = null; resetSession(false); });
        document.getElementById('pl-chaos').addEventListener('click', function () {
            activePreset = null;
            FORCE_SCALE = 4 + Math.random() * 10;
            FRICTION = 0.18 + Math.random() * 0.22;
            RMAX = 55 + Math.random() * 65;
            REPULSION_SCALE = 1.2 + Math.random() * 1.2;
            resetSession(false);
            syncSlider('pl-force', Math.round(FORCE_SCALE));
            syncSlider('pl-friction', Math.round(FRICTION * 100));
            syncSlider('pl-range', Math.round(RMAX));
            syncSlider('pl-repulsion', Math.round(REPULSION_SCALE * 100));
        });

        document.getElementById('pl-share').addEventListener('click', function () {
            var shareBtn = this;
            var hash = buildShareHash();
            var url = window.location.origin + window.location.pathname + hash;
            try { history.replaceState(null, '', hash); } catch (e) { }
            var done = function () {
                shareBtn.textContent = 'Copied!';
                setTimeout(function () { shareBtn.textContent = 'Share'; }, 1600);
            };
            if (window.copyWithToast) {
                window.copyWithToast(url, 'Link copied!', done);
            } else if (navigator.clipboard) {
                navigator.clipboard.writeText(url).then(done);
            }
        });
        document.getElementById('pl-export').addEventListener('click', exportConfig);
        document.getElementById('pl-import').addEventListener('change', function (e) {
            if (e.target.files && e.target.files[0]) {
                importConfig(e.target.files[0]);
                e.target.value = '';
            }
        });

        document.getElementById('pl-mode-simple').addEventListener('click', function () { setEditorMode('simple'); });
        document.getElementById('pl-mode-advanced').addEventListener('click', function () { setEditorMode('advanced'); });
        document.getElementById('pl-mode-presets').addEventListener('click', function () { setEditorMode('presets'); });

        document.getElementById('pl-presets-view').addEventListener('click', function (e) {
            var btn = e.target.closest('.pl-preset-btn');
            if (!btn) return;
            var preset = PRESETS[parseInt(btn.dataset.preset, 10)];
            if (preset) applyConfig(preset, preset.name);
        });

        document.getElementById('pl-lc-enabled').addEventListener('change', function (e) {
            lifecycle.enabled = e.target.checked;
            sendUpdate('lifecycle', lifecycle);
            var lcCtrl = document.getElementById('pl-lc-controls');
            if (lcCtrl) lcCtrl.style.display = lifecycle.enabled ? 'flex' : 'none';
        });
        var lcCtrl = document.getElementById('pl-lc-controls');
        if (lcCtrl) lcCtrl.style.display = lifecycle.enabled ? 'flex' : 'none';

        bind('pl-lc-repro', function (v) { lifecycle.reproRate = v / 100; sendUpdate('lifecycle', lifecycle); });
        bind('pl-lc-mutation', function (v) { lifecycle.mutationRate = v / 100; sendUpdate('lifecycle', lifecycle); });
        bind('pl-lc-lifespan', function (v) { lifecycle.lifespan = v; sendUpdate('lifecycle', lifecycle); });
        bind('pl-lc-egain', function (v) { lifecycle.energyGain = v / 100; sendUpdate('lifecycle', lifecycle); });
        bind('pl-lc-ecost', function (v) { lifecycle.energyCost = v / 100; sendUpdate('lifecycle', lifecycle); });
        bind('pl-lc-predation', function (v) { lifecycle.predationStr = v / 100; sendUpdate('lifecycle', lifecycle); });
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
        var pBtn = document.getElementById('pl-mode-presets');
        if (sBtn) sBtn.classList.toggle('active', mode === 'simple');
        if (aBtn) aBtn.classList.toggle('active', mode === 'advanced');
        if (pBtn) pBtn.classList.toggle('active', mode === 'presets');

        var knobs = document.getElementById('pl-knobs-view');
        var presets = document.getElementById('pl-presets-view');
        if (knobs) knobs.style.display = mode === 'presets' ? 'none' : 'block';
        if (presets) presets.style.display = mode === 'presets' ? 'grid' : 'none';

        if (mode === 'advanced') buildSpeciesPanel();
        else removeSpeciesPanel();
    }

    function buildSpeciesPanel() {
        removeSpeciesPanel();
        speciesPanel = document.createElement('div');
        speciesPanel.className = 'pl-species-panel';
        speciesPanel.addEventListener('input', function (e) {
            if (!e.target.classList.contains('pl-sp-slider')) return;
            var si = parseInt(e.target.dataset.i, 10);
            var sj = parseInt(e.target.dataset.j, 10);
            var field = e.target.dataset.field;
            var v = parseFloat(e.target.value);
            var idx = si * NUM_SPECIES + sj;
            if (field === 'attraction') {
                currentAttractions[idx] = v / 100;
                sendUpdate('attractions', currentAttractions);
            } else if (field === 'minDist') {
                currentMinDist[idx] = v / 100;
                sendUpdate('minDist', currentMinDist);
            }
        });
        rebuildSpeciesPanel();
        document.body.appendChild(speciesPanel);
        makeDraggable(speciesPanel);
    }

    function rebuildSpeciesPanel() {
        if (!speciesPanel || !currentAttractions) return;
        var n = NUM_SPECIES;
        var palette = getPalette();
        var existingHandle = speciesPanel.querySelector('.pl-drag-handle');

        var html = '<div class="pl-sp-header">Species Editor</div><div class="pl-sp-scroll">';
        html += '<div class="pl-sp-section">Attraction Matrix</div>';
        html += '<div class="pl-sp-note">Row \u2192 Column. Positive = attract, Negative = repel.</div>';
        html += '<table class="pl-sp-matrix"><thead><tr><th></th>';
        for (var j = 0; j < n; j++) {
            html += '<th><span class="pl-sp-dot" style="background:' + paletteToHex(palette[j]) + '"></span>' + speciesNames[j] + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (var i = 0; i < n; i++) {
            html += '<tr><td><span class="pl-sp-dot" style="background:' + paletteToHex(palette[i]) + '"></span>' + speciesNames[i] + '</td>';
            for (var j2 = 0; j2 < n; j2++) {
                var val = Math.round(currentAttractions[i * n + j2] * 100);
                html += '<td><input type="range" min="-100" max="100" value="' + val + '" class="pl-sp-slider" data-i="' + i + '" data-j="' + j2 + '" data-field="attraction"></td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        html += '<div class="pl-sp-section">Min Distance Matrix</div>';
        html += '<table class="pl-sp-matrix"><thead><tr><th></th>';
        for (var j3 = 0; j3 < n; j3++) html += '<th>' + speciesNames[j3] + '</th>';
        html += '</tr></thead><tbody>';
        for (var i2 = 0; i2 < n; i2++) {
            html += '<tr><td>' + speciesNames[i2] + '</td>';
            for (var j4 = 0; j4 < n; j4++) {
                var mval = Math.round(currentMinDist[i2 * n + j4] * 100);
                html += '<td><input type="range" min="5" max="60" value="' + mval + '" class="pl-sp-slider" data-i="' + i2 + '" data-j="' + j4 + '" data-field="minDist"></td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table></div>';

        if (existingHandle) {
            var wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            while (speciesPanel.childNodes.length > 1) speciesPanel.removeChild(speciesPanel.lastChild);
            while (wrapper.firstChild) speciesPanel.appendChild(wrapper.firstChild);
        } else {
            speciesPanel.innerHTML = html;
        }
    }

    function removeSpeciesPanel() {
        cleanupPanel(speciesPanel);
        speciesPanel = null;
    }

    function removeControls() {
        cleanupPanel(controlsPanel);
        controlsPanel = null;
    }

    function slugifyPreset(name) {
        return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    }

    function buildShareHash() {
        if (activePreset) return '#pl=' + slugifyPreset(activePreset);
        var n = NUM_SPECIES, att = [], md = [];
        for (var i = 0; i < n * n; i++) {
            att.push(Math.round(currentAttractions[i] * 100));
            md.push(Math.round(currentMinDist[i] * 100));
        }
        var payload = {
            s: n, n: NUM_PARTICLES, r: Math.round(RMAX),
            f: Math.round(FRICTION * 100), d: Math.round(DT * 1000),
            k: Math.round(FORCE_SCALE), p: Math.round(REPULSION_SCALE * 100),
            w: WALL_MODE, a: att, m: md
        };
        if (lifecycle.enabled) payload.l = lifecycle;
        var b64 = btoa(JSON.stringify(payload)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
        return '#pl=cfg:' + b64;
    }

    function parseShareHash() {
        var h = window.location.hash || '';
        if (h.indexOf('#pl=') !== 0) return null;
        var val = h.slice(4);
        if (val.indexOf('cfg:') === 0) {
            try {
                var b64 = val.slice(4).replace(/-/g, '+').replace(/_/g, '/');
                while (b64.length % 4) b64 += '=';
                var o = JSON.parse(atob(b64));
                return {
                    name: null,
                    cfg: {
                        numSpecies: o.s, numParticles: o.n, rmax: o.r,
                        friction: o.f / 100, dt: o.d / 1000, forceScale: o.k,
                        repulsionScale: o.p / 100, wallMode: o.w,
                        attractions: o.a.map(function (x) { return x / 100; }),
                        minDist: o.m.map(function (x) { return x / 100; }),
                        lifecycle: o.l || { enabled: false }
                    }
                };
            } catch (e) { return null; }
        }
        for (var i = 0; i < PRESETS.length; i++) {
            if (slugifyPreset(PRESETS[i].name) === val) {
                return { name: PRESETS[i].name, cfg: PRESETS[i] };
            }
        }
        return null;
    }

    function enterImmersive(initialShare) {
        immersive = true;
        running = true;
        document.body.classList.add('immersive-mode');
        canvas.style.pointerEvents = 'auto';
        if (initialShare) {
            applyConfig(initialShare.cfg, initialShare.name);
        } else {
            resetSession(false);
            buildControls();
            if (editorMode === 'advanced') buildSpeciesPanel();
        }
        animId = requestAnimationFrame(render);
    }

    function exitImmersive() {
        immersive = false;
        running = false;
        pointerMode = 0;
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        document.body.classList.remove('immersive-mode');
        canvas.style.pointerEvents = 'none';
        removeControls();
        removeSpeciesPanel();
        if (worker) { worker.terminate(); worker = null; }
        displayBuf = null;
        displayView = null;
        displayN = 0;
        recycleBuf = null;
        if ((window.location.hash || '').indexOf('#pl=') === 0) {
            try { history.replaceState(null, '', window.location.pathname + window.location.search); } catch (e) { }
        }
    }

    var btn = document.getElementById('immersive-toggle');
    if (btn) {
        btn.innerHTML = plIconSvg;
        btn.addEventListener('click', function () {
            if (immersive) exitImmersive();
            else enterImmersive(null);
        });

        setInterval(function () {
            if (!immersive && btn) {
                btn.classList.add('pl-btn-glow');
                setTimeout(function () { btn.classList.remove('pl-btn-glow'); }, 2000);
            }
        }, 30000);
    }

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && immersive) exitImmersive();
    });

    var shared = parseShareHash();
    if (shared) {
        enterImmersive(shared);
    }
})();
