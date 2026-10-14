# Verification

With the development server running:

```sh
npm run test:neural-field
npm run test:steering
node tests/site-browser.mjs
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
