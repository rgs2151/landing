FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html stylesheet.css 404.html CNAME favicon.ico .nojekyll /usr/share/nginx/html/
COPY images /usr/share/nginx/html/images
COPY scripts /usr/share/nginx/html/scripts

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1/ || exit 1
