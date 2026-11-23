var rngState, numParticles, numSpecies, worldW, worldH, halfW, halfH;
var rmax, rmaxSq, invRmax, friction, dt, forceScale;
var attractions, minDist, invMinDist, invOneMinusMd;
var gridCols, gridRows, gridCells, invCellX, invCellY;
var cellCount, cellStart, cellCursor, pCell, sortedIdx;
var sPosX, sPosY, sSpecies;
var posX, posY, velX, velY, speciesArr;
var fxBuf, fyBuf;
var nStart = new Int32Array(9), nEnd = new Int32Array(9);
var nOffX = new Float32Array(9), nOffY = new Float32Array(9);
var bounce = 0.3, stepScale, frictionFactor;
var repulsionScale = 1.0;
var wallMode = 1;
var maxParticles = 40000;
var MAX_SAMPLE = 36;

var energy, age, alive;
var deadStack, deadTop = 0;
var reproRate = 0.0, mutationRate = 0.0, lifespan = 0;
var energyGain = 0.0, energyCost = 0.0, predationStr = 0.0;
var lifecycleOn = false;

var mouseX = 0, mouseY = 0, mouseMode = 0;

var FLOATS_PER_PARTICLE = 4;
var spawnQueue = [];
var spawnBudget = 120;

function xorshift32() {
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    return (rngState >>> 0) / 4294967295;
}

function precomputeMinDist() {
    var len = minDist.length;
    if (!invMinDist || invMinDist.length < len) {
        invMinDist = new Float32Array(len);
        invOneMinusMd = new Float32Array(len);
    }
    for (var i = 0; i < len; i++) {
        var mr = minDist[i];
        if (mr < 0.01) mr = 0.01;
        else if (mr > 0.99) mr = 0.99;
        invMinDist[i] = 1 / mr;
        invOneMinusMd[i] = 1 / (1 - mr);
    }
}

function allocGrid() {
    var cs = rmax < 1 ? 1 : rmax;
    gridCols = Math.max(1, Math.floor(worldW / cs));
    gridRows = Math.max(1, Math.floor(worldH / cs));
    invCellX = gridCols / worldW;
    invCellY = gridRows / worldH;
    gridCells = gridCols * gridRows;
    if (!cellCount || cellCount.length < gridCells) {
        cellCount = new Int32Array(gridCells);
        cellStart = new Int32Array(gridCells + 1);
        cellCursor = new Int32Array(gridCells);
    }
}

function buildGrid() {
    var i, gx, gy, ci, sum, dst;
    cellCount.fill(0, 0, gridCells);

    for (i = 0; i < numParticles; i++) {
        if (!alive[i]) {
            pCell[i] = -1;
            continue;
        }
        gx = (posX[i] * invCellX) | 0;
        gy = (posY[i] * invCellY) | 0;
        if (gx < 0) gx = 0;
        else if (gx >= gridCols) gx = gridCols - 1;
        if (gy < 0) gy = 0;
        else if (gy >= gridRows) gy = gridRows - 1;
        ci = gy * gridCols + gx;
        pCell[i] = ci;
        cellCount[ci]++;
    }

    sum = 0;
    for (i = 0; i < gridCells; i++) {
        cellStart[i] = sum;
        cellCursor[i] = sum;
        sum += cellCount[i];
    }
    cellStart[gridCells] = sum;

    for (i = 0; i < numParticles; i++) {
        ci = pCell[i];
        if (ci < 0) continue;
        dst = cellCursor[ci]++;
        sortedIdx[dst] = i;
        sPosX[dst] = posX[i];
        sPosY[dst] = posY[i];
        sSpecies[dst] = speciesArr[i];
    }
}

function simulate() {
    buildGrid();

    var wrap = wallMode === 1;
    var cols = gridCols, rows = gridRows;
    var wW = worldW, wH = worldH, hW = halfW, hH = halfH;
    var rSq = rmaxSq, invR = invRmax, rep = repulsionScale;
    var nSp = numSpecies;
    var lc = lifecycleOn, pred = predationStr, predDistSq = (rmax * 0.3) * (rmax * 0.3);
    var att = attractions, md = minDist, invMd = invMinDist, inv1m = invOneMinusMd;
    var maxS = MAX_SAMPLE;

    var gy, gx, ci, cStart, cEnd, nn, dy, dx, ny, nx, ox, oy, nyCols, nci;
    var s, nb, start, end, cnt, stride, off, qx, qy, k;
    var px, py, rowOff, fx, fy, lfx, lfy, ddx, ddy, distSq, dist, r, mr, f, invD, idx, orig;

    for (gy = 0; gy < rows; gy++) {
        for (gx = 0; gx < cols; gx++) {
            ci = gy * cols + gx;
            cStart = cellStart[ci];
            cEnd = cellStart[ci + 1];
            if (cStart === cEnd) continue;

            nn = 0;
            for (dy = -1; dy <= 1; dy++) {
                ny = gy + dy;
                oy = 0;
                if (ny < 0) {
                    if (!wrap) continue;
                    ny += rows;
                    oy = -wH;
                } else if (ny >= rows) {
                    if (!wrap) continue;
                    ny -= rows;
                    oy = wH;
                }
                nyCols = ny * cols;
                for (dx = -1; dx <= 1; dx++) {
                    nx = gx + dx;
                    ox = 0;
                    if (nx < 0) {
                        if (!wrap) continue;
                        nx += cols;
                        ox = -wW;
                    } else if (nx >= cols) {
                        if (!wrap) continue;
                        nx -= cols;
                        ox = wW;
                    }
                    nci = nyCols + nx;
                    if (cellStart[nci] < cellStart[nci + 1]) {
                        nStart[nn] = cellStart[nci];
                        nEnd[nn] = cellStart[nci + 1];
                        nOffX[nn] = ox;
                        nOffY[nn] = oy;
                        nn++;
                    }
                }
            }

            for (s = cStart; s < cEnd; s++) {
                px = sPosX[s];
                py = sPosY[s];
                rowOff = sSpecies[s] * nSp;
                orig = sortedIdx[s];
                fx = 0;
                fy = 0;

                for (nb = 0; nb < nn; nb++) {
                    start = nStart[nb];
                    end = nEnd[nb];
                    cnt = end - start;
                    stride = cnt > maxS ? ((cnt + maxS - 1) / maxS) | 0 : 1;
                    off = stride > 1 ? (s % stride) : 0;
                    qx = px - nOffX[nb];
                    qy = py - nOffY[nb];
                    lfx = 0;
                    lfy = 0;

                    for (k = start + off; k < end; k += stride) {
                        ddx = sPosX[k] - qx;
                        ddy = sPosY[k] - qy;
                        distSq = ddx * ddx + ddy * ddy;
                        if (distSq === 0 || distSq >= rSq) continue;
                        dist = Math.sqrt(distSq);
                        r = dist * invR;
                        idx = rowOff + sSpecies[k];
                        mr = md[idx];
                        if (r < mr) {
                            f = (r * invMd[idx] - 1) * rep;
                        } else {
                            f = att[idx] * (1 - Math.abs(2 * r - 1 - mr) * inv1m[idx]);
                        }
                        invD = f / dist;
                        lfx += ddx * invD;
                        lfy += ddy * invD;

                        if (lc && distSq < predDistSq && pred > 0) {
                            var aij = att[idx];
                            if (aij > 0.2) {
                                var steal = pred * 0.002 * aij * stride;
                                energy[orig] += steal;
                                energy[sortedIdx[k]] -= steal * 0.5;
                            }
                        }
                    }
                    fx += lfx * stride;
                    fy += lfy * stride;
                }
                fxBuf[orig] = fx;
                fyBuf[orig] = fy;
            }
        }
    }

    var forceStep = rmax * forceScale * dt * 0.05;
    var maxVel = (rmax * 0.35) / Math.max(stepScale, 0.1);
    var maxVelSq = maxVel * maxVel;
    var mActive = mouseMode !== 0;
    var mRad = rmax * 2.5, mRadSq = mRad * mRad;
    var mStr = mouseMode * 12.0;

    for (var i = 0; i < numParticles; i++) {
        if (!alive[i]) continue;
        fx = fxBuf[i];
        fy = fyBuf[i];

        if (mActive) {
            ddx = mouseX - posX[i];
            ddy = mouseY - posY[i];
            if (wrap) {
                if (ddx > hW) ddx -= wW;
                else if (ddx < -hW) ddx += wW;
                if (ddy > hH) ddy -= wH;
                else if (ddy < -hH) ddy += wH;
            }
            distSq = ddx * ddx + ddy * ddy;
            if (distSq > 1 && distSq < mRadSq) {
                dist = Math.sqrt(distSq);
                var mf = mStr * (1 - dist / mRad) / dist;
                fx += ddx * mf;
                fy += ddy * mf;
            }
        }

        var nvx = (velX[i] + fx * forceStep) * frictionFactor;
        var nvy = (velY[i] + fy * forceStep) * frictionFactor;
        var vSq = nvx * nvx + nvy * nvy;
        if (vSq > maxVelSq) {
            var vScale = maxVel / Math.sqrt(vSq);
            nvx *= vScale;
            nvy *= vScale;
        }
        velX[i] = nvx;
        velY[i] = nvy;
        posX[i] += nvx * stepScale;
        posY[i] += nvy * stepScale;

        if (wrap) {
            if (posX[i] < 0) posX[i] += wW;
            else if (posX[i] >= wW) posX[i] -= wW;
            if (posY[i] < 0) posY[i] += wH;
            else if (posY[i] >= wH) posY[i] -= wH;
        } else {
            if (posX[i] < 0) { posX[i] = 0; velX[i] *= -bounce; }
            else if (posX[i] > wW) { posX[i] = wW; velX[i] *= -bounce; }
            if (posY[i] < 0) { posY[i] = 0; velY[i] *= -bounce; }
            else if (posY[i] > wH) { posY[i] = wH; velY[i] *= -bounce; }
        }

        if (lc) {
            var speedSq = velX[i] * velX[i] + velY[i] * velY[i];
            energy[i] += energyGain * 0.01;
            energy[i] -= energyCost * 0.001 * (1 + Math.sqrt(speedSq) * 0.1);
            age[i]++;

            if (lifespan > 0 && age[i] > lifespan) {
                energy[i] -= 0.02 * (age[i] - lifespan) / lifespan;
            }

            if (energy[i] <= 0) {
                alive[i] = 0;
                deadStack[deadTop++] = i;
                continue;
            }
            if (energy[i] > 1.5 && reproRate > 0 && spawnQueue.length < spawnBudget) {
                if (deadTop > 0 || numParticles < maxParticles) {
                    if (xorshift32() < reproRate * 0.005) {
                        var childSpecies = speciesArr[i];
                        if (mutationRate > 0 && xorshift32() < mutationRate * 0.1) {
                            childSpecies = (xorshift32() * nSp) | 0;
                            if (childSpecies >= nSp) childSpecies = nSp - 1;
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
    }

    if (lc && spawnQueue.length > 0) {
        processSpawnQueue();
    }
}

function processSpawnQueue() {
    var count = Math.min(spawnQueue.length, spawnBudget);
    for (var q = 0; q < count; q++) {
        var slot = deadTop > 0 ? deadStack[--deadTop] : (numParticles < maxParticles ? numParticles++ : -1);
        if (slot < 0) break;
        var child = spawnQueue[q];
        posX[slot] = ((child.x % worldW) + worldW) % worldW;
        posY[slot] = ((child.y % worldH) + worldH) % worldH;
        velX[slot] = child.vx;
        velY[slot] = child.vy;
        speciesArr[slot] = child.sp;
        energy[slot] = 0.8;
        age[slot] = 0;
        alive[slot] = 1;
    }
    spawnQueue.length = 0;
}

function writeFrame(ab) {
    var needed = numParticles * FLOATS_PER_PARTICLE * 4;
    if (!ab || ab.byteLength < needed) {
        ab = new ArrayBuffer(Math.max(needed, maxParticles * FLOATS_PER_PARTICLE * 4));
    }
    var f32 = new Float32Array(ab);
    var i, base, written = 0;
    if (lifecycleOn) {
        var invLife = 1 / Math.max(lifespan, 1);
        for (i = 0; i < numParticles; i++) {
            if (!alive[i]) continue;
            base = written << 2;
            f32[base] = posX[i];
            f32[base + 1] = posY[i];
            f32[base + 2] = speciesArr[i];
            var nrg = energy[i];
            var ageF = age[i] * invLife;
            var dim = nrg < 0.25 ? 0.25 : (nrg > 1.0 ? 1.0 : nrg);
            if (ageF > 0.8) dim *= Math.max(0.25, 1.0 - (ageF - 0.8) * 2.0);
            f32[base + 3] = dim;
            written++;
        }
    } else {
        for (i = 0; i < numParticles; i++) {
            if (!alive[i]) continue;
            base = written << 2;
            f32[base] = posX[i];
            f32[base + 1] = posY[i];
            f32[base + 2] = speciesArr[i];
            f32[base + 3] = 1.0;
            written++;
        }
    }
    return { buf: ab, n: written };
}

function stepAndPost(ab) {
    simulate();
    var res = writeFrame(ab);
    self.postMessage({ type: 'frame', buf: res.buf, n: res.n }, [res.buf]);
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
    pCell = new Int32Array(cap);
    sortedIdx = new Int32Array(cap);
    sPosX = new Float32Array(cap);
    sPosY = new Float32Array(cap);
    sSpecies = new Uint8Array(cap);
    deadStack = new Int32Array(cap);
    deadTop = 0;
}

function initParticles(count) {
    for (var i = 0; i < count; i++) {
        posX[i] = xorshift32() * worldW;
        posY[i] = xorshift32() * worldH;
        velX[i] = (xorshift32() - 0.5) * 2;
        velY[i] = (xorshift32() - 0.5) * 2;
        var sp = (xorshift32() * numSpecies) | 0;
        speciesArr[i] = sp >= numSpecies ? numSpecies - 1 : sp;
        energy[i] = 0.5 + xorshift32() * 0.5;
        age[i] = 0;
        alive[i] = 1;
    }
}

function applyLifecycle(lc) {
    if (lc.enabled !== undefined) lifecycleOn = !!lc.enabled;
    if (lc.reproRate !== undefined) reproRate = lc.reproRate;
    if (lc.mutationRate !== undefined) mutationRate = lc.mutationRate;
    if (lc.lifespan !== undefined) lifespan = lc.lifespan;
    if (lc.energyGain !== undefined) energyGain = lc.energyGain;
    if (lc.energyCost !== undefined) energyCost = lc.energyCost;
    if (lc.predationStr !== undefined) predationStr = lc.predationStr;
}

function onInit(cfg) {
    numParticles = cfg.numParticles | 0;
    numSpecies = cfg.numSpecies | 0;
    worldW = cfg.worldW;
    worldH = cfg.worldH;
    halfW = worldW * 0.5;
    halfH = worldH * 0.5;
    rmax = cfg.rmax;
    friction = cfg.friction;
    dt = cfg.dt;
    forceScale = cfg.forceScale;
    attractions = cfg.attractions;
    minDist = cfg.minDist;
    precomputeMinDist();
    rngState = cfg.seed >>> 0;
    if (cfg.repulsionScale !== undefined) repulsionScale = cfg.repulsionScale;
    if (cfg.wallMode !== undefined) wallMode = cfg.wallMode;
    if (cfg.bounce !== undefined) bounce = cfg.bounce;
    if (cfg.lifecycle !== undefined) applyLifecycle(cfg.lifecycle);

    rmaxSq = rmax * rmax;
    invRmax = 1 / rmax;
    stepScale = dt * 60;
    frictionFactor = Math.pow(1 - friction, dt / 0.01);

    var cap = Math.max(numParticles, maxParticles);
    allocArrays(cap);
    initParticles(numParticles);
    allocGrid();
    spawnQueue.length = 0;
    mouseMode = 0;

    var bufSize = cap * FLOATS_PER_PARTICLE * 4;
    stepAndPost(new ArrayBuffer(bufSize));
}

self.onmessage = function (e) {
    var d = e.data;
    if (d.type === 'init') {
        onInit(d.config);
        return;
    }
    if (d.type === 'step') {
        stepAndPost(d.buf);
        return;
    }
    if (d.type === 'mouse') {
        mouseX = d.x;
        mouseY = d.y;
        mouseMode = d.mode;
        return;
    }
    if (d.type === 'update') {
        if (d.friction !== undefined) {
            friction = d.friction;
            frictionFactor = Math.pow(1 - friction, dt / 0.01);
        }
        if (d.forceScale !== undefined) forceScale = d.forceScale;
        if (d.rmax !== undefined) {
            rmax = d.rmax;
            rmaxSq = rmax * rmax;
            invRmax = 1 / rmax;
            allocGrid();
        }
        if (d.dt !== undefined) {
            dt = d.dt;
            stepScale = dt * 60;
            frictionFactor = Math.pow(1 - friction, dt / 0.01);
        }
        if (d.bounce !== undefined) bounce = d.bounce;
        if (d.repulsionScale !== undefined) repulsionScale = d.repulsionScale;
        if (d.wallMode !== undefined) wallMode = d.wallMode;
        if (d.attractions !== undefined) attractions = d.attractions;
        if (d.minDist !== undefined) {
            minDist = d.minDist;
            precomputeMinDist();
        }
        if (d.lifecycle !== undefined) applyLifecycle(d.lifecycle);
        if (d.resize !== undefined) {
            worldW = d.resize.worldW;
            worldH = d.resize.worldH;
            halfW = worldW * 0.5;
            halfH = worldH * 0.5;
            for (var i = 0; i < numParticles; i++) {
                if (posX[i] >= worldW) posX[i] = posX[i] % worldW;
                if (posY[i] >= worldH) posY[i] = posY[i] % worldH;
            }
            allocGrid();
        }
    }
};
