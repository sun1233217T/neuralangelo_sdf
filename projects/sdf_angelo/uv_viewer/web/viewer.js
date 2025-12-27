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
  lines.push(`Render FPS: ${formatNumber(renderFps, 1)}`);
  if (lastStats) {
    if (lastStats.update_fps !== undefined) {
      lines.push(`Texture FPS: ${formatNumber(lastStats.update_fps, 2)}`);
    }
    if (lastStats.last_ms !== undefined) {
      lines.push(`Texture ms: ${formatNumber(lastStats.last_ms, 1)}`);
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
    setCamera: async (camera) => {
      await fetch("/api/camera", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera),
      });
    },
  };
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

  texture = new THREE.Texture();
  if (texture.colorSpace !== undefined && THREE.SRGBColorSpace) {
    texture.colorSpace = THREE.SRGBColorSpace;
  }
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
  viewerEl.addEventListener("mousedown", (event) => {
    controls.dragging = true;
    if (controls.mode === "trackball") {
      controls.dragStart = projectOnTrackball(event.clientX, event.clientY);
    }
  });
  window.addEventListener("mouseup", () => {
    controls.dragging = false;
    controls.dragStart = null;
  });
  window.addEventListener("mousemove", (event) => {
    if (!controls.dragging) {
      return;
    }
    if (controls.mode === "trackball") {
      const curr = projectOnTrackball(event.clientX, event.clientY);
      trackballRotate(curr);
    }
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
