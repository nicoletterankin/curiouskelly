FROM node:20-bullseye AS builder

WORKDIR /workspace

RUN corepack enable

COPY package.json pnpm-workspace.yaml tsconfig.base.json .editorconfig .prettierrc.json ./
COPY packages ./packages
COPY services ./services
COPY apps ./apps

RUN pnpm install --no-frozen-lockfile
RUN pnpm --filter @acme/gateway build
RUN pnpm --filter @acme/classroom build

FROM node:20-bullseye-slim AS runner

ENV NODE_ENV=production
WORKDIR /app

RUN corepack enable

COPY --from=builder /workspace/package.json .
COPY --from=builder /workspace/pnpm-workspace.yaml .
COPY --from=builder /workspace/packages ./packages
COPY --from=builder /workspace/services ./services
COPY --from=builder /workspace/apps/gateway/dist ./apps/gateway/dist

# Install only runtime deps for gateway
RUN pnpm install --prod --filter @acme/gateway

EXPOSE 8080

ENV PORT=8080

CMD ["pnpm", "--filter", "@acme/gateway", "start"]

