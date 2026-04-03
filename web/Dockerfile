# Container image definition for the PuntLab Next.js dashboard.
# Scope: installs dependencies, performs a production build, and starts the
# web server inside a Node 20 Alpine image.
# Dependencies: `package.json`, `package-lock.json`, `next.config.js`, and the
# `app/` plus `lib/` source trees.

FROM node:20-alpine AS build

WORKDIR /app

COPY package.json ./package.json
COPY package-lock.json ./package-lock.json
COPY tsconfig.json ./tsconfig.json
COPY next-env.d.ts ./next-env.d.ts
COPY next.config.js ./next.config.js
COPY eslint.config.mjs ./eslint.config.mjs
COPY app ./app
COPY lib ./lib

ENV NEXT_TELEMETRY_DISABLED=1

RUN npm ci \
    && npm run build

FROM node:20-alpine AS runtime

WORKDIR /app

COPY package.json ./package.json
COPY package-lock.json ./package-lock.json
COPY next.config.js ./next.config.js
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/app ./app
COPY --from=build /app/lib ./lib

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

EXPOSE 3000

CMD ["npm", "run", "start"]
