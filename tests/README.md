# Verification

With the development server running:

```sh
npm run test:neural-field
npm run test:steering
node tests/site-browser.mjs
node tests/mobile-browser.mjs
node tests/mobile-video-compat.mjs
node tests/site-arrival-browser.mjs
node tests/neuraljax-brain-browser.mjs
node tests/steering-browser-parity.mjs
```

The browser scripts use local Playwright/Chromium tooling under
`/tmp/landing-browser-check` and `/home/dev/.cache/ms-playwright`.
They inspect DOM, geometry, pixels and behavior without screenshots or performance
benchmarks. `fixtures/approved-site.json` records F's accepted geometry and copy
at promotion; update only after an explicitly requested design/content change.

For the production build, run `npm run build` and `npm run preview`, then set
`SITE_URL` to its URL for `site-browser.mjs` and `site-arrival-browser.mjs`.
The two source-parity tests require the development server for module imports.
Old variant-specific tests can be recovered from Git history.

The mobile check covers touch navigation, 320-844px widths including landscape,
all frozen project copy/links, figure enlargement, browser history, returning to
the same collage scroll position, playback controls, and switching to desktop.
The desktop baseline remains unchanged. These are browser-emulated checks, not
claims of testing on physical iOS or Android devices.

The mobile checks also cover staggered inline figures and text collision at four
widths. The compatibility test simulates a decoder that discards WebM alpha,
then verifies all eleven H.264 fallbacks animate with transparency, preserve
source hashes/dimensions/durations, and respect pause/replay. It requires FFprobe.
This does not substitute for testing Brave on a physical iPhone.
