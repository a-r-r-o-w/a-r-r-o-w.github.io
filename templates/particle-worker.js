var rngState, numParticles, numSpecies, worldW, worldH;
var rmax, rmaxSq, invRmax, friction, dt, forceScale;
var attractions, minDist, leftZoneEnd, rightZoneStart;
var cellSize, invCell, gridCols, gridRows, gridCells;
var grid, gridCounts, maxPerCell = 24;
var posX, posY, velX, velY, speciesArr;
var fxBuf, fyBuf;
var simTime, outBuf, intervalId;
var bounce = 0.3, stepScale, frictionFactor;
var frameBufRef, frameF32;
var repulsionScale = 1.0, velDamping = 1.0;
var wallMode = 1;
var maxParticles = 40000;

var energy, age, alive;
var reproRate = 0.0, mutationRate = 0.0, lifespan = 0;
var energyGain = 0.0, energyCost = 0.0, predationStr = 0.0;
var lifecycleOn = false;

var FLOATS_PER_PARTICLE = 7;
var spawnQueue = [];
var spawnBudget = 120;
var tickCount = 0;

function xorshift32() {
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    return (rngState >>> 0) / 4294967295;
}

function forceFn(r, a, mr) {
    if (r < mr) {
        return (r / mr - 1) * repulsionScale;
    }
    if (r < 1) {
        return a * (1 - Math.abs(2 * r - 1 - mr) / (1 - mr));
    }
    return 0;
}

function allocGrid() {
    cellSize = rmax;
    if (cellSize < 1) {
        cellSize = 1;
    }
    invCell = 1 / cellSize;
    gridCols = ((worldW / cellSize) | 0) + 3;
    gridRows = ((worldH / cellSize) | 0) + 3;
    gridCells = gridCols * gridRows;
    if (!grid || grid.length < gridCells * maxPerCell) {
        grid = new Int32Array(gridCells * maxPerCell);
        gridCounts = new Int32Array(gridCells);
    }
}

function buildGrid() {
    var i, gx, gy, ci;
    for (i = 0; i < gridCells; i++) {
        gridCounts[i] = 0;
    }
    for (i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            continue;
        }
        gx = (posX[i] * invCell) | 0;
        gy = (posY[i] * invCell) | 0;
        if (gx < 0) {
            gx = 0;
        }
        else if (gx >= gridCols) {
            gx = gridCols - 1;
        }
        if (gy < 0) {
            gy = 0;
        }
        else if (gy >= gridRows) {
            gy = gridRows - 1;
        }
        ci = gy * gridCols + gx;
        if (gridCounts[ci] < maxPerCell) {
            grid[ci * maxPerCell + gridCounts[ci]] = i;
            gridCounts[ci]++;
        }
    }
}

function simulate() {
    buildGrid();
    var i, j, k, px, py, si, gx, gy;
    var dx, dy, nx, ny, ci, cnt;
    var ddx, ddy, distSq, dist, r, f, invD, idx;
    var midGap, forceStep;

    for (i = 0; i < numParticles; i++) {
        fxBuf[i] = 0;
        fyBuf[i] = 0;
    }

    for (i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            continue;
        }
        px = posX[i];
        py = posY[i];
        si = speciesArr[i];
        gx = (px * invCell) | 0;
        gy = (py * invCell) | 0;
        if (gx < 0) {
            gx = 0;
        }
        else if (gx >= gridCols) {
            gx = gridCols - 1;
        }
        if (gy < 0) {
            gy = 0;
        }
        else if (gy >= gridRows) {
            gy = gridRows - 1;
        }

        for (dy = -1; dy <= 1; dy++) {
            ny = gy + dy;
            if (ny < 0 || ny >= gridRows) {
                continue;
            }
            for (dx = -1; dx <= 1; dx++) {
                nx = gx + dx;
                if (nx < 0 || nx >= gridCols) {
                    continue;
                }
                ci = ny * gridCols + nx;
                cnt = gridCounts[ci];
                for (k = 0; k < cnt; k++) {
                    j = grid[ci * maxPerCell + k];
                    if (j === i || !alive[j]) {
                        continue;
                    }
                    ddx = posX[j] - px;
                    ddy = posY[j] - py;
                    distSq = ddx * ddx + ddy * ddy;
                    if (distSq === 0 || distSq >= rmaxSq) {
                        continue;
                    }
                    dist = Math.sqrt(distSq);
                    r = dist * invRmax;
                    idx = si * numSpecies + speciesArr[j];
                    f = forceFn(r, attractions[idx], minDist[idx]);
                    invD = f / dist;
                    fxBuf[i] += ddx * invD;
                    fyBuf[i] += ddy * invD;

                    if (lifecycleOn && dist < rmax * 0.3) {
                        var aij = attractions[idx];
                        if (aij > 0.2 && predationStr > 0) {
                            var steal = predationStr * 0.002 * aij;
                            energy[i] += steal;
                            energy[j] -= steal * 0.5;
                        }
                    }
                }
            }
        }
    }

    midGap = leftZoneEnd + rightZoneStart;
    forceStep = rmax * forceScale * dt;

    for (i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            continue;
        }
        velX[i] = (velX[i] + fxBuf[i] * forceStep) * frictionFactor * velDamping;
        velY[i] = (velY[i] + fyBuf[i] * forceStep) * frictionFactor * velDamping;
        posX[i] += velX[i] * stepScale;
        posY[i] += velY[i] * stepScale;

        if (posY[i] < 0) {
            posY[i] = wallMode === 1 ? worldH + posY[i] : 0;
            velY[i] *= wallMode === 1 ? 1 : -bounce;
        }
        else if (posY[i] > worldH) {
            posY[i] = wallMode === 1 ? posY[i] - worldH : worldH;
            velY[i] *= wallMode === 1 ? 1 : -bounce;
        }

        if (leftZoneEnd < rightZoneStart) {
            px = posX[i];
            if (px > leftZoneEnd && px < rightZoneStart) {
                if (px * 2 < midGap) {
                    posX[i] = leftZoneEnd;
                    velX[i] = -Math.abs(velX[i]) * bounce;
                }
                else {
                    posX[i] = rightZoneStart;
                    velX[i] = Math.abs(velX[i]) * bounce;
                }
            }
        }

        if (posX[i] < 0) {
            posX[i] = wallMode === 1 ? worldW + posX[i] : 0;
            velX[i] *= wallMode === 1 ? 1 : -bounce;
        }
        else if (posX[i] > worldW) {
            posX[i] = wallMode === 1 ? posX[i] - worldW : worldW;
            velX[i] *= wallMode === 1 ? 1 : -bounce;
        }

        if (lifecycleOn) {
            var speed = Math.sqrt(velX[i] * velX[i] + velY[i] * velY[i]);
            energy[i] += energyGain * 0.01;
            energy[i] -= energyCost * 0.001 * (1 + speed * 0.1);
            age[i]++;

            if (lifespan > 0 && age[i] > lifespan) {
                energy[i] -= 0.02 * (age[i] - lifespan) / lifespan;
            }

            if (energy[i] <= 0) {
                alive[i] = 0;
                continue;
            }
            if (energy[i] > 1.5 && reproRate > 0 && numParticles < maxParticles && spawnQueue.length < spawnBudget) {
                if (xorshift32() < reproRate * 0.005) {
                    var childSpecies = speciesArr[i];
                    if (mutationRate > 0 && xorshift32() < mutationRate * 0.1) {
                        childSpecies = (xorshift32() * numSpecies) | 0;
                        if (childSpecies >= numSpecies) {
                            childSpecies = numSpecies - 1;
                        }
                    }
                    spawnQueue.push({
                        x: posX[i] + (xorshift32() - 0.5) * 10,
                        y: posY[i] + (xorshift32() - 0.5) * 10,
                        vx: velX[i] * 0.3 + (xorshift32() - 0.5) * 0.5,
                        vy: velY[i] * 0.3 + (xorshift32() - 0.5) * 0.5,
                        sp: childSpecies
                    });
                    energy[i] *= 0.5;
                }
            }
        }
    }

    if (lifecycleOn && spawnQueue.length > 0) {
        processSpawnQueue();
    }
}

function processSpawnQueue() {
    var count = Math.min(spawnQueue.length, spawnBudget);
    for (var q = 0; q < count; q++) {
        var child = spawnQueue[q];
        var slot = findDeadSlot();
        if (slot < 0) {
            break;
        }
        posX[slot] = child.x;
        posY[slot] = child.y;
        velX[slot] = child.vx;
        velY[slot] = child.vy;
        speciesArr[slot] = child.sp;
        energy[slot] = 0.8;
        age[slot] = 0;
        alive[slot] = 1;
    }
    spawnQueue.length = 0;
}

function findDeadSlot() {
    for (var i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            return i;
        }
    }
    if (numParticles < maxParticles) {
        return numParticles++;
    }
    return -1;
}

function writeFrame(ab) {
    if (frameBufRef !== ab) {
        frameBufRef = ab;
        frameF32 = new Float32Array(ab);
    }
    var f32 = frameF32;
    var i, base, vx, vy, written = 0;
    for (i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            continue;
        }
        base = written * FLOATS_PER_PARTICLE;
        vx = velX[i];
        vy = velY[i];
        f32[base] = posX[i];
        f32[base + 1] = posY[i];
        f32[base + 2] = Math.sqrt(vx * vx + vy * vy);
        f32[base + 3] = speciesArr[i];
        f32[base + 4] = Math.sqrt(fxBuf[i] * fxBuf[i] + fyBuf[i] * fyBuf[i]);
        f32[base + 5] = lifecycleOn ? energy[i] : 1.0;
        f32[base + 6] = lifecycleOn ? age[i] / Math.max(lifespan, 1) : 0.0;
        written++;
    }
    return written;
}

function tick() {
    if (!outBuf) {
        return;
    }
    tickCount++;
    simTime += dt;
    simulate();
    var needed = numParticles * FLOATS_PER_PARTICLE * 4;
    if (outBuf.byteLength < needed) {
        outBuf = new ArrayBuffer(needed);
    }
    var n = writeFrame(outBuf);
    var b = outBuf;
    outBuf = null;
    self.postMessage({ type: 'frame', buf: b, n: n }, [b]);
}

function placeInZone() {
    var lw = leftZoneEnd;
    var rw = worldW - rightZoneStart;
    if (lw + rw <= 0) {
        return xorshift32() * worldW;
    }
    var t = xorshift32() * (lw + rw);
    if (t < lw) {
        return t;
    }
    return rightZoneStart + (t - lw);
}

function initParticles(count) {
    var i, px, py;
    for (i = 0; i < count; i++) {
        if (leftZoneEnd < rightZoneStart) {
            px = placeInZone();
        }
        else {
            px = xorshift32() * worldW;
        }
        py = xorshift32() * worldH;

        posX[i] = px;
        posY[i] = py;
        velX[i] = (xorshift32() - 0.5) * 2;
        velY[i] = (xorshift32() - 0.5) * 2;
        speciesArr[i] = (xorshift32() * numSpecies) | 0;
        if (speciesArr[i] >= numSpecies) {
            speciesArr[i] = numSpecies - 1;
        }
        energy[i] = 0.5 + xorshift32() * 0.5;
        age[i] = 0;
        alive[i] = 1;
    }
}

function allocArrays(cap) {
    posX = new Float32Array(cap);
    posY = new Float32Array(cap);
    velX = new Float32Array(cap);
    velY = new Float32Array(cap);
    speciesArr = new Uint8Array(cap);
    fxBuf = new Float32Array(cap);
    fyBuf = new Float32Array(cap);
    energy = new Float32Array(cap);
    age = new Float32Array(cap);
    alive = new Uint8Array(cap);
}

function onInit(cfg) {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }

    numParticles = cfg.numParticles | 0;
    numSpecies = cfg.numSpecies | 0;
    worldW = cfg.worldW;
    worldH = cfg.worldH;
    rmax = cfg.rmax;
    friction = cfg.friction;
    dt = cfg.dt;
    forceScale = cfg.forceScale;
    attractions = cfg.attractions;
    minDist = cfg.minDist;
    leftZoneEnd = cfg.leftZoneEnd;
    rightZoneStart = cfg.rightZoneStart;
    rngState = cfg.seed >>> 0;
    if (cfg.repulsionScale !== undefined) {
        repulsionScale = cfg.repulsionScale;
    }
    if (cfg.velDamping !== undefined) {
        velDamping = cfg.velDamping;
    }
    if (cfg.wallMode !== undefined) {
        wallMode = cfg.wallMode;
    }
    if (cfg.lifecycle !== undefined) {
        applyLifecycle(cfg.lifecycle);
    }

    rmaxSq = rmax * rmax;
    invRmax = 1 / rmax;
    stepScale = dt * 60;
    frictionFactor = Math.pow(1 - friction, dt * 60);

    var cap = Math.max(numParticles, maxParticles);
    allocArrays(cap);
    initParticles(numParticles);
    allocGrid();
    simTime = 0;
    tickCount = 0;
    spawnQueue.length = 0;
    outBuf = new ArrayBuffer(cap * FLOATS_PER_PARTICLE * 4);
    intervalId = setInterval(tick, 1000 / 30);
}

function applyLifecycle(lc) {
    if (lc.enabled !== undefined) {
        lifecycleOn = !!lc.enabled;
    }
    if (lc.reproRate !== undefined) {
        reproRate = lc.reproRate;
    }
    if (lc.mutationRate !== undefined) {
        mutationRate = lc.mutationRate;
    }
    if (lc.lifespan !== undefined) {
        lifespan = lc.lifespan;
    }
    if (lc.energyGain !== undefined) {
        energyGain = lc.energyGain;
    }
    if (lc.energyCost !== undefined) {
        energyCost = lc.energyCost;
    }
    if (lc.predationStr !== undefined) {
        predationStr = lc.predationStr;
    }
}

self.onmessage = function (e) {
    var d = e.data;
    if (d.type === 'init') {
        onInit(d.config);
        return;
    }
    if (d.type === 'step') {
        outBuf = d.buf;
        return;
    }
    if (d.type === 'update') {
        if (d.friction !== undefined) {
            friction = d.friction;
            frictionFactor = Math.pow(1 - friction, dt * 60);
        }
        if (d.forceScale !== undefined) {
            forceScale = d.forceScale;
        }
        if (d.rmax !== undefined) {
            rmax = d.rmax;
            rmaxSq = rmax * rmax;
            invRmax = 1 / rmax;
            allocGrid();
        }
        if (d.dt !== undefined) {
            dt = d.dt;
            stepScale = dt * 60;
            frictionFactor = Math.pow(1 - friction, dt * 60);
        }
        if (d.bounce !== undefined) {
            bounce = d.bounce;
        }
        if (d.repulsionScale !== undefined) {
            repulsionScale = d.repulsionScale;
        }
        if (d.velDamping !== undefined) {
            velDamping = d.velDamping;
        }
        if (d.wallMode !== undefined) {
            wallMode = d.wallMode;
        }
        if (d.zones !== undefined) {
            leftZoneEnd = d.zones.left;
            rightZoneStart = d.zones.right;
        }
        if (d.attractions !== undefined) {
            attractions = d.attractions;
        }
        if (d.minDist !== undefined) {
            minDist = d.minDist;
        }
        if (d.lifecycle !== undefined) {
            applyLifecycle(d.lifecycle);
        }
    }
};
