let renderer;
let scene;
let camera;
let mesh;
let texture;

let updateMs = 1000;
let texturePending = false;
let textureUpdates = 0;
let initialized = false;
let api = null;
let renderFrames = 0;
let renderFps = 0;
let lastRenderTime = performance.now();
let lastStats = null;
let textureTransport = "png";
let textureSize = 1024;

const statusEl = document.getElementById("status");
const viewerEl = document.getElementById("viewer");

const controls = {
  mode: "trackball",
  invert: true,
  radius: 2,
  target: null,
  offset: null,
  up: null,
  dragging: false,
  dragStart: null,
  pointers: new Map(),
  lastPinchDist: null,
};

function setStatus(text) {
  if (statusEl) {
    statusEl.textContent = text;
  }
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function updateStatus(note) {
  const lines = [];
  if (note) {
    lines.push(note);
  }
  const invertLabel = controls.invert ? " (invert)" : "";
  lines.push(`Mode: ${controls.mode}${invertLabel}`);
  lines.push(`Transport: ${textureTransport}`);
  lines.push(`Render FPS: ${formatNumber(renderFps, 1)}`);
  if (lastStats) {
    if (lastStats.update_fps !== undefined) {
      lines.push(`Update FPS: ${formatNumber(lastStats.update_fps, 2)}`);
    }
    if (lastStats.render_ms !== undefined) {
      lines.push(`Render ms: ${formatNumber(lastStats.render_ms, 1)}`);
    }
    if (lastStats.encode_ms !== undefined) {
      lines.push(`Encode ms: ${formatNumber(lastStats.encode_ms, 1)}`);
    }
    if (lastStats.total_ms !== undefined) {
      lines.push(`Total ms: ${formatNumber(lastStats.total_ms, 1)}`);
    }
    if (lastStats.gpu_allocated_mb !== undefined) {
      const alloc = formatNumber(lastStats.gpu_allocated_mb, 0);
      const reserved = formatNumber(lastStats.gpu_reserved_mb, 0);
      lines.push(`GPU MB: ${alloc} / ${reserved}`);
    }
  }
  lines.push(`Texture updates: ${textureUpdates}`);
  setStatus(lines.join("\n"));
}

function createApi() {
  if (window.pywebview && window.pywebview.api) {
    const api = window.pywebview.api;
    return {
      mode: "pywebview",
      getInit: () => api.get_init(),
      getTexture: () => api.get_texture(),
      setCamera: (camera) => api.set_camera(camera),
      getTextureRaw: null,
    };
  }
  return {
    mode: "http",
    getInit: async () => {
      const res = await fetch("/api/init");
      return res.json();
    },
    getTexture: async () => {
      const res = await fetch("/api/texture");
      return res.json();
    },
    getTextureRaw: async () => {
      const res = await fetch("/api/texture_raw");
      if (!res.ok || res.status === 204) {
        return null;
      }
      const buffer = await res.arrayBuffer();
      const stats = parseStatsFromHeaders(res.headers);
      return { buffer, stats };
    },
    setCamera: async (camera) => {
      await fetch("/api/camera", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera),
      });
    },
  };
}

function parseStatsFromHeaders(headers) {
  const stats = {};
  const updateFps = parseFloat(headers.get("X-Update-Fps"));
  if (!Number.isNaN(updateFps)) {
    stats.update_fps = updateFps;
  }
  const renderMs = parseFloat(headers.get("X-Render-Ms"));
  if (!Number.isNaN(renderMs)) {
    stats.render_ms = renderMs;
  }
  const encodeMs = parseFloat(headers.get("X-Encode-Ms"));
  if (!Number.isNaN(encodeMs)) {
    stats.encode_ms = encodeMs;
  }
  const totalMs = parseFloat(headers.get("X-Total-Ms"));
  if (!Number.isNaN(totalMs)) {
    stats.total_ms = totalMs;
  }
  const gpuAlloc = parseFloat(headers.get("X-GPU-Allocated-MB"));
  if (!Number.isNaN(gpuAlloc)) {
    stats.gpu_allocated_mb = gpuAlloc;
  }
  const gpuReserved = parseFloat(headers.get("X-GPU-Reserved-MB"));
  if (!Number.isNaN(gpuReserved)) {
    stats.gpu_reserved_mb = gpuReserved;
  }
  const gpuMax = parseFloat(headers.get("X-GPU-Max-Allocated-MB"));
  if (!Number.isNaN(gpuMax)) {
    stats.gpu_max_allocated_mb = gpuMax;
  }
  return stats;
}

function parseOBJ(text) {
  const positions = [];
  const normals = [];
  const uvs = [];
  const outPos = [];
  const outNorm = [];
  const outUV = [];
  const indices = [];
  const vertMap = new Map();

  const lines = text.split("\n");
  for (let raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const parts = line.split(/\s+/);
    const tag = parts[0];
    if (tag === "v") {
      positions.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3]),
      ]);
    } else if (tag === "vt") {
      uvs.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
      ]);
    } else if (tag === "vn") {
      normals.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3]),
      ]);
    } else if (tag === "f") {
      const face = parts.slice(1).map((token) => getIndex(token));
      for (let i = 1; i + 1 < face.length; i += 1) {
        indices.push(face[0], face[i], face[i + 1]);
      }
    }
  }

  function parseIndex(value, length) {
    if (!value) {
      return null;
    }
    let idx = parseInt(value, 10);
    if (Number.isNaN(idx)) {
      return null;
    }
    if (idx < 0) {
      idx = length + idx;
    } else {
      idx -= 1;
    }
    return idx;
  }

  function getIndex(token) {
    if (vertMap.has(token)) {
      return vertMap.get(token);
    }
    const comps = token.split("/");
    const vIdx = parseIndex(comps[0], positions.length);
    const vtIdx = parseIndex(comps[1], uvs.length);
    const vnIdx = parseIndex(comps[2], normals.length);

    const pos = vIdx !== null ? positions[vIdx] : [0, 0, 0];
    outPos.push(pos[0], pos[1], pos[2]);

    if (vtIdx !== null && uvs[vtIdx]) {
      const uv = uvs[vtIdx];
      outUV.push(uv[0], uv[1]);
    } else {
      outUV.push(0, 0);
    }

    if (vnIdx !== null && normals[vnIdx]) {
      const n = normals[vnIdx];
      outNorm.push(n[0], n[1], n[2]);
    } else {
      outNorm.push(0, 0, 0);
    }

    const idx = outPos.length / 3 - 1;
    vertMap.set(token, idx);
    return idx;
  }

  return {
    positions: new Float32Array(outPos),
    normals: new Float32Array(outNorm),
    uvs: new Float32Array(outUV),
    indices: new Uint32Array(indices),
  };
}

function buildGeometry(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(data.uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));

  let hasNormals = false;
  for (let i = 0; i < data.normals.length; i += 1) {
    if (data.normals[i] !== 0) {
      hasNormals = true;
      break;
    }
  }
  if (hasNormals) {
    geometry.setAttribute("normal", new THREE.BufferAttribute(data.normals, 3));
  } else {
    geometry.computeVertexNormals();
  }
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function initRenderer() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
  if (renderer.outputColorSpace !== undefined && THREE.SRGBColorSpace) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  }
  renderer.domElement.style.touchAction = "none";
  viewerEl.appendChild(renderer.domElement);
}

function initScene(geometry) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0f12);

  camera = new THREE.PerspectiveCamera(
    45,
    viewerEl.clientWidth / viewerEl.clientHeight,
    0.01,
    1000
  );

  texture = createTexture();
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    side: THREE.DoubleSide,
  });

  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  const box = new THREE.Box3().setFromObject(mesh);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxSize = Math.max(size.x, size.y, size.z);
  controls.target = center;
  controls.radius = maxSize * 2.2 || 2.0;
  controls.offset = new THREE.Vector3();
  controls.up = new THREE.Vector3(0, 1, 0);
  const theta = Math.PI / 4;
  const phi = Math.PI / 2.2;
  const x = controls.radius * Math.sin(phi) * Math.sin(theta);
  const y = controls.radius * Math.cos(phi);
  const z = controls.radius * Math.sin(phi) * Math.cos(theta);
  controls.offset.set(x, y, z);
  updateCameraFromOffset();
}

function createTexture() {
  if (textureTransport === "raw_rgba") {
    const data = new Uint8Array(textureSize * textureSize * 4);
    const tex = new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );
    tex.flipY = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    if (tex.colorSpace !== undefined && THREE.SRGBColorSpace) {
      tex.colorSpace = THREE.SRGBColorSpace;
    }
    tex.needsUpdate = true;
    return tex;
  }
  const tex = new THREE.Texture();
  if (tex.colorSpace !== undefined && THREE.SRGBColorSpace) {
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  return tex;
}

function updateCameraFromOffset() {
  if (!controls.offset || !controls.up) {
    return;
  }
  const minRadius = 0.1;
  const maxRadius = 10000.0;
  const r = controls.offset.length();
  if (r < minRadius) {
    controls.offset.setLength(minRadius);
  } else if (r > maxRadius) {
    controls.offset.setLength(maxRadius);
  }
  controls.radius = controls.offset.length();
  controls.up.normalize();
  camera.position.copy(controls.target).add(controls.offset);
  camera.up.copy(controls.up);
  camera.lookAt(controls.target);
}

function onResize() {
  if (!renderer || !camera) {
    return;
  }
  const w = viewerEl.clientWidth;
  const h = viewerEl.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function projectOnTrackball(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = 1 - ((clientY - rect.top) / rect.height) * 2;
  const len2 = x * x + y * y;
  let z = 0;
  if (len2 <= 1) {
    z = Math.sqrt(1 - len2);
  } else {
    const norm = 1 / Math.sqrt(len2);
    return new THREE.Vector3(x * norm, y * norm, 0);
  }
  return new THREE.Vector3(x, y, z);
}

function rotateCamera(axis, angle) {
  if (!controls.offset || !controls.up) {
    return;
  }
  if (axis.lengthSq() < 1e-8) {
    return;
  }
  const quat = new THREE.Quaternion().setFromAxisAngle(axis.normalize(), angle);
  controls.offset.applyQuaternion(quat);
  controls.up.applyQuaternion(quat);
  updateCameraFromOffset();
}

function trackballRotate(curr) {
  if (!controls.dragStart) {
    controls.dragStart = curr;
    return;
  }
  const start = controls.dragStart;
  const axisCam = new THREE.Vector3().crossVectors(start, curr);
  const dot = THREE.MathUtils.clamp(start.dot(curr), -1, 1);
  const angle = Math.acos(dot);
  if (axisCam.lengthSq() < 1e-8 || angle === 0) {
    return;
  }
  const axisWorld = axisCam.clone().applyQuaternion(camera.quaternion);
  const rotSign = controls.invert ? -1 : 1;
  rotateCamera(axisWorld, angle * rotSign);
  controls.dragStart = curr;
}

function attachControls() {
  const canvas = renderer.domElement;
  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    controls.pointers.set(event.pointerId, {
      x: event.clientX,
      y: event.clientY,
      type: event.pointerType,
    });
    const touchPointers = getTouchPointers();
    if (event.pointerType === "touch" && touchPointers.length === 2) {
      controls.dragging = false;
      controls.dragStart = null;
      controls.lastPinchDist = pinchDistance(touchPointers[0], touchPointers[1]);
      return;
    }
    controls.dragging = true;
    if (controls.mode === "trackball") {
      controls.dragStart = projectOnTrackball(event.clientX, event.clientY);
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!controls.pointers.has(event.pointerId)) {
      return;
    }
    controls.pointers.set(event.pointerId, {
      x: event.clientX,
      y: event.clientY,
      type: event.pointerType,
    });
    if (event.pointerType === "touch") {
      const touchPointers = getTouchPointers();
      if (touchPointers.length === 2) {
        const dist = pinchDistance(touchPointers[0], touchPointers[1]);
        if (controls.lastPinchDist && controls.offset) {
          const scale = controls.lastPinchDist / dist;
          controls.offset.multiplyScalar(scale);
          updateCameraFromOffset();
        }
        controls.lastPinchDist = dist;
        return;
      }
    }
    if (!controls.dragging) {
      return;
    }
    if (controls.mode === "trackball") {
      const curr = projectOnTrackball(event.clientX, event.clientY);
      trackballRotate(curr);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (controls.pointers.has(event.pointerId)) {
      controls.pointers.delete(event.pointerId);
    }
    const touchPointers = getTouchPointers();
    if (touchPointers.length < 2) {
      controls.lastPinchDist = null;
    }
    if (touchPointers.length === 1) {
      const t = touchPointers[0];
      controls.dragging = true;
      controls.dragStart = projectOnTrackball(t.x, t.y);
    } else {
      controls.dragging = false;
      controls.dragStart = null;
    }
  });
  canvas.addEventListener("pointercancel", (event) => {
    if (controls.pointers.has(event.pointerId)) {
      controls.pointers.delete(event.pointerId);
    }
    controls.dragging = false;
    controls.dragStart = null;
    controls.lastPinchDist = null;
  });

  viewerEl.addEventListener("wheel", (event) => {
    event.preventDefault();
    const delta = Math.sign(event.deltaY);
    if (controls.offset) {
      controls.offset.multiplyScalar(1.0 + delta * 0.08);
      updateCameraFromOffset();
    }
  }, { passive: false });

  window.addEventListener("keydown", (event) => {
    if (!controls.offset || !controls.up) {
      return;
    }
    const key = event.key.toLowerCase();
    const rotateStep = event.shiftKey ? 0.12 : 0.06;
    const rollStep = event.shiftKey ? 0.12 : 0.06;
    const zoomStep = event.shiftKey ? 0.86 : 0.93;
    const rotSign = controls.invert ? -1 : 1;
    const viewDir = controls.offset.clone().normalize();
    const rightDir = new THREE.Vector3().crossVectors(controls.up, viewDir).normalize();
    if (key === "i") {
      rotateCamera(rightDir, rotateStep * rotSign);
    } else if (key === "k") {
      rotateCamera(rightDir, -rotateStep * rotSign);
    } else if (key === "j") {
      rotateCamera(controls.up.clone(), rotateStep * rotSign);
    } else if (key === "l") {
      rotateCamera(controls.up.clone(), -rotateStep * rotSign);
    } else if (key === "u") {
      rotateCamera(viewDir, rollStep * rotSign);
    } else if (key === "o") {
      rotateCamera(viewDir, -rollStep * rotSign);
    } else if (key === "=" || key === "+") {
      controls.offset.multiplyScalar(zoomStep);
      updateCameraFromOffset();
    } else if (key === "-" || key === "_") {
      controls.offset.multiplyScalar(1.0 / zoomStep);
      updateCameraFromOffset();
    } else if (key === "r") {
      controls.up.set(0, 1, 0);
      controls.offset.set(controls.radius * 0.7, controls.radius * 0.6, controls.radius * 0.9);
      updateCameraFromOffset();
    }
  });
}

function getTouchPointers() {
  const touches = [];
  for (const value of controls.pointers.values()) {
    if (value.type === "touch") {
      touches.push(value);
    }
  }
  return touches;
}

function pinchDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

let lastCamSent = new THREE.Vector3();
let lastCamSendTime = 0;

async function sendCameraIfNeeded() {
  if (!camera || !api) {
    return;
  }
  const now = performance.now();
  if (now - lastCamSendTime < 120) {
    return;
  }
  const pos = camera.position;
  if (pos.distanceToSquared(lastCamSent) < 1e-6) {
    return;
  }
  lastCamSent.copy(pos);
  lastCamSendTime = now;
  await api.setCamera({ x: pos.x, y: pos.y, z: pos.z });
}

function animate() {
  requestAnimationFrame(animate);
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
    renderFrames += 1;
    const now = performance.now();
    if (now - lastRenderTime >= 1000) {
      renderFps = (renderFrames * 1000) / (now - lastRenderTime);
      renderFrames = 0;
      lastRenderTime = now;
      updateStatus();
    }
    sendCameraIfNeeded();
  }
}

async function pollTexture() {
  if (texturePending) {
    return;
  }
  if (!api) {
    return;
  }
  texturePending = true;
  try {
    if (textureTransport === "raw_rgba" && api.getTextureRaw) {
      const result = await api.getTextureRaw();
      if (result && result.stats) {
        lastStats = result.stats;
      }
      if (result && result.buffer) {
        const data = new Uint8Array(result.buffer);
        const expected = textureSize * textureSize * 4;
        if (data.length === expected && texture.image && texture.image.data) {
          if (texture.image.data.length !== data.length) {
            texture.image.data = data;
          } else {
            texture.image.data.set(data);
          }
          texture.needsUpdate = true;
          textureUpdates += 1;
          updateStatus();
        } else if (data.length !== expected) {
          updateStatus(`Raw size mismatch: ${data.length} vs ${expected}`);
        }
      }
    } else {
      const result = await api.getTexture();
      let dataUrl = null;
      if (typeof result === "string") {
        dataUrl = result;
      } else if (result && typeof result === "object") {
        dataUrl = result.data_url || null;
        if (result.stats) {
          lastStats = result.stats;
        }
      }
      if (dataUrl) {
        const img = new Image();
        img.onload = () => {
          texture.image = img;
          texture.needsUpdate = true;
          textureUpdates += 1;
          updateStatus();
        };
        img.src = dataUrl;
      }
    }
  } catch (err) {
    updateStatus(`Texture error: ${err}`);
  } finally {
    texturePending = false;
  }
}

async function init() {
  if (initialized) {
    return;
  }
  initialized = true;
  api = createApi();
  setStatus("Loading mesh...");
  const initData = await api.getInit();
  updateMs = initData.update_ms || updateMs;
  textureSize = initData.texture_size || textureSize;
  textureTransport = initData.texture_transport || textureTransport;

  const meshData = parseOBJ(initData.mesh_obj);
  const geometry = buildGeometry(meshData);

  initRenderer();
  initScene(geometry);
  attachControls();
  window.addEventListener("resize", onResize);

  setStatus("Viewer ready.");
  animate();
  await pollTexture();
  setInterval(pollTexture, updateMs);
}

window.addEventListener("pywebviewready", init);
window.addEventListener("DOMContentLoaded", () => {
  if (window.location.protocol.startsWith("http")) {
    init();
  }
});
