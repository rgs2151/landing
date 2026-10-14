import { readFileSync, writeFileSync } from 'node:fs';

// Offline generation only: preserve the accepted seeded landscapes exactly.
const base = new URL('../design/references/robust-steering-animation/', import.meta.url);
const examples = JSON.parse(readFileSync(new URL('mgsm-examples.json', base))).examples;
const document = {
  createElementNS() {
    return { attributes: {}, dataset: {}, children: [],
      setAttribute(key, value) { this.attributes[key] = value; },
      replaceChildren(...children) { this.children = children; },
    };
  },
};
  function makeStateContours(node, seed, components, color, fractions = [.04, .08, .14, .23, .34, .47, .61, .76, .9]) {
    // Seeded Gaussian-mixture samples and a Gaussian KDE, not hand-drawn ellipses.
    // Synthetic illustration only: these are not recorded model activations.
    const random = () => {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    const samples = Array.from({ length: 600 }, () => {
      let draw = random() * totalWeight;
      const c = components.find((component) => (draw -= component.weight) < 0) || components[components.length - 1];
      const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-10)));
      const theta = 2 * Math.PI * random();
      const a = radius * Math.cos(theta);
      const b = radius * Math.sin(theta);
      return [c.x + c.sx * a, c.y + c.sy * (c.rho * a + Math.sqrt(1 - c.rho * c.rho) * b)];
    });
    // Evaluate beyond the visible square, so contours can cross its edges cleanly.
    const n = 128;
    const density = new Float64Array(n * n);
    const bandwidth = .45;
    let peak = 0;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const px = -6 + x * 12 / (n - 1);
        const py = 6 - y * 12 / (n - 1);
        let value = 0;
        for (const [sx, sy] of samples) {
          const distance = ((px - sx) ** 2 + (py - sy) ** 2) / (2 * bandwidth * bandwidth);
          if (distance < 18) value += Math.exp(-distance);
        }
        density[y * n + x] = value;
        peak = Math.max(peak, value);
      }
    }
    const cases = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []];
    const paths = [];
    for (const fraction of fractions) {
      const threshold = peak * fraction;
      const vertices = new Map();
      const adjacency = new Map();
      const segments = [];
      for (let y = 0; y < n - 1; y++) {
        for (let x = 0; x < n - 1; x++) {
          const values = [density[y * n + x], density[y * n + x + 1], density[(y + 1) * n + x + 1], density[(y + 1) * n + x]];
          const mask = values.reduce((code, v, i) => code | (v >= threshold ? 1 << i : 0), 0);
          if (!cases[mask].length) continue;
          const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
          const keys = ["h:" + x + ":" + y, "v:" + (x + 1) + ":" + y, "h:" + x + ":" + (y + 1), "v:" + x + ":" + y];
          const vertex = (edge) => {
            const key = keys[edge];
            if (!vertices.has(key)) {
              const next = (edge + 1) % 4;
              const t = (threshold - values[edge]) / (values[next] - values[edge]);
              const [ax, ay] = corners[edge];
              const [bx, by] = corners[next];
              vertices.set(key, [-70 + (ax + (bx - ax) * t) * 480 / (n - 1), -82 + (ay + (by - ay) * t) * 480 / (n - 1)]);
            }
            return key;
          };
          for (const [a, b] of cases[mask]) {
            const pair = [vertex(a), vertex(b)];
            const index = segments.push(pair) - 1;
            for (const key of pair) {
              if (!adjacency.has(key)) adjacency.set(key, []);
              adjacency.get(key).push(index);
            }
          }
        }
      }
      const used = new Set();
      let d = "";
      segments.forEach(([start, end], index) => {
        if (used.has(index)) return;
        const points = [vertices.get(start), vertices.get(end)];
        used.add(index);
        let key = end;
        while (key !== start) {
          const nextIndex = adjacency.get(key).find((i) => !used.has(i));
          if (nextIndex === undefined) break;
          used.add(nextIndex);
          const pair = segments[nextIndex];
          key = pair[0] === key ? pair[1] : pair[0];
          points.push(vertices.get(key));
        }
        // Closed rings get a very light density fill; open boundary contours stay unfilled.
        if (key !== start) return;
        d += "M" + points.map(([x, y]) => x.toFixed(2) + "," + y.toFixed(2)).join("L") + "Z";
      });
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", color);
      path.setAttribute("fill-opacity", ".055");
      path.setAttribute("fill-rule", "evenodd");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-opacity", ".6");
      path.setAttribute("stroke-width", "1.1");
      path.setAttribute("stroke-linejoin", "round");
      paths.push(path);
    }
    node.replaceChildren(...paths);
    node.dataset.sampleCount = samples.length;
    node.dataset.method = "Seeded Gaussian-mixture sampling; Gaussian KDE; marching-squares contours";
  }

  // Compute once, then reuse exactly the same fields for every replay.
  // The multi-modal landscapes are independent of the two language-direction points.
  const englishComponents = [
    { weight: .25, x: 0, y: 0, sx: 2.6, sy: 2.6, rho: -.1 },
    { weight: .13, x: -2.4, y: 1.5, sx: .75, sy: .9, rho: -.3 },
    { weight: .1, x: -.45, y: 2.2, sx: 1.0, sy: .55, rho: -.2 },
    { weight: .13, x: 2.0, y: 1.5, sx: .8, sy: .8, rho: .2 },
    { weight: .12, x: -1.75, y: -.7, sx: .9, sy: .65, rho: .1 },
    { weight: .1, x: .7, y: .1, sx: .75, sy: .9, rho: -.4 },
    { weight: .08, x: 2.5, y: -1.5, sx: .7, sy: .8, rho: -.2 },
    { weight: .09, x: -.4, y: -2.4, sx: 1.15, sy: .55, rho: .1 },
  ];
  const shiftedComponents = [
    { weight: .3, x: 0, y: 0, sx: 2.9, sy: 2.9, rho: .08 },
    { weight: .13, x: -1.7, y: 2.1, sx: .85, sy: 1.0, rho: .5 },
    { weight: .16, x: 1.2, y: -1.6, sx: 1.1, sy: .8, rho: .5 },
    { weight: .1, x: .7, y: 1.3, sx: .8, sy: .55, rho: -.45 },
    { weight: .07, x: -1.6, y: -1.7, sx: .9, sy: .5, rho: -.2 },
    { weight: .06, x: 2.8, y: 1.8, sx: .55, sy: .9, rho: -.3 },
    { weight: .05, x: -3.1, y: -.1, sx: .7, sy: .85, rho: .2 },
    { weight: .04, x: -.2, y: -.2, sx: .4, sy: .4, rho: -.1 },
    { weight: .035, x: -2.9, y: 2.5, sx: .4, sy: .4, rho: .2 },
    { weight: .03, x: 2.1, y: -3, sx: .7, sy: .45, rho: .15 },
    { weight: .025, x: 2.7, y: -.1, sx: .45, sy: .6, rho: -.2 },
  ];

function contourRoute(paths, seed) {
  // Sample a seeded start on an undisplayed level set, then choose a distant endpoint.
  // Following the dense marching-squares vertices keeps the route on that level set.
  const candidates = [];
  for (const path of paths) {
    for (const match of path.d.matchAll(/M([^Z]+)Z/g)) {
      const ring = match[1].split('L').map(point => point.split(',').map(Number));
      if (ring.length < 20) continue;
      ring.pop();
      const start = (seed * 97) % ring.length;
      const ordered = [...ring.slice(start), ...ring.slice(0, start)];
      let run = [];
      const save = () => {
        if (run.length < 20) return;
        const distance = ([x,y]) => Math.hypot(x - run[0][0], y - run[0][1]);
        const end = run.reduce((best, point, i) => distance(point) > distance(run[best]) ? i : best, 0);
        if (distance(run[end]) >= 130) candidates.push(run.slice(0, end + 1));
      };
      for (const point of ordered) {
        const [x,y] = point;
        if (x < 44 || x > 296 || y < 32 || y > 284) { save(); run = []; }
        else run.push(point);
      }
      save();
    }
  }
  if (!candidates.length) throw new Error('No sufficiently separated contour route for seed ' + seed);
  const points = candidates[seed % candidates.length];
  const end = points.at(-1), before = points.at(-2);
  return {
    d: 'M' + points.map(point => point.join(',')).join('L'),
    start: points[0], end,
    angle: Math.atan2(end[1] - before[1], end[0] - before[0]) * 180 / Math.PI,
  };
}

const fields = examples.map((entry, index) => {
  const node = document.createElementNS();
  const english = entry.language === 'en';
  makeStateContours(node, 2027 + index, english ? englishComponents : shiftedComponents, english ? '#919191' : '#8B1E1E');
  const paths = node.children.map(child => child.attributes);
  const hidden = document.createElementNS();
  if (!english) makeStateContours(hidden, 2027 + index, shiftedComponents, '#8B1E1E', [.06, .11, .185, .285, .405, .54, .685, .83]);
  return { language: entry.language, paths, route: english ? null : contourRoute(hidden.children.map(child => child.attributes), 2027 + index) };
});
writeFileSync(new URL('steering-landscapes.json', base), JSON.stringify({
  note: 'Synthetic KDE illustration, not measured model trajectories. Generated by scripts/cache-steering-landscapes.mjs.',
  fields,
}) + '\n');
console.log('Cached ' + fields.length + ' landscapes and five contour-following routes.');

