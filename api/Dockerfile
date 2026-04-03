# Container build for the PuntLab Fastify API.
# Scope: installs the API dependencies, compiles the TypeScript server, and runs
# the production Node.js process in a Node 20 Alpine image.
# Dependencies: `package.json`, `tsconfig.json`, and the `src/` tree.

FROM node:20-alpine AS build

WORKDIR /app

COPY package.json ./package.json
COPY package-lock.json ./package-lock.json
COPY tsconfig.json ./tsconfig.json
COPY src ./src

ENV PRISMA_SKIP_POSTINSTALL_GENERATE=true

RUN npm ci \
    && npm run build

FROM node:20-alpine AS runtime

WORKDIR /app

COPY package.json ./package.json
COPY package-lock.json ./package-lock.json
COPY tsconfig.json ./tsconfig.json
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
