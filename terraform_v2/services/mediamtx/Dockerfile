FROM bluenviron/mediamtx:latest
COPY mediamtx.yml /mediamtx.yml
 
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q --spider http://localhost:8889/ || exit 1
 
USER 65534