import * as THREE from "three";

export function createScene(appElement) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x6fb7ff);
  scene.fog = new THREE.Fog(0x6fb7ff, 200, 900);

  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    2000,
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  if (appElement) {
    appElement.appendChild(renderer.domElement);
  }

  // Lights
  const ambientLight = new THREE.AmbientLight(0x6b8cff, 0.6);
  scene.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xf5f7ff, 1.0);
  sunLight.position.set(50, 120, 40);
  sunLight.castShadow = true;
  scene.add(sunLight);

  // Water plane
  const waterGeom = new THREE.PlaneGeometry(4000, 4000, 1, 1);
  const waterMat = new THREE.MeshPhongMaterial({
    color: 0x064a7a,
    shininess: 70,
    specular: 0x5bb0ff,
  });
  const water = new THREE.Mesh(waterGeom, waterMat);
  water.rotation.x = -Math.PI / 2;
  water.receiveShadow = true;
  scene.add(water);

  // 1 m x 1 m grid
  const gridSize = 400;
  const gridDivisions = 400;
  const grid = new THREE.GridHelper(
    gridSize,
    gridDivisions,
    0x2aa3ff,
    0x1b4b72,
  );
  grid.position.y = 0.02;
  scene.add(grid);

  return { scene, camera, renderer };
}

export function handleResize(camera, renderer) {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

