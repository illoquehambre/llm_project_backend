FROM node:22.14.0 as builder

WORKDIR /app

COPY package*.json ./
COPY tailwind.config.cjs ./
COPY tsconfig.json ./
COPY astro.config.mjs ./
COPY components.json ./

RUN npm install

COPY src ./src
COPY public ./public

RUN npm run build

FROM node:22.14.0
RUN npm install -g serve

WORKDIR /app
COPY --from=builder /app/dist ./dist

EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]