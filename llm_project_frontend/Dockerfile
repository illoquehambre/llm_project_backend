# Etapa de construcción
FROM node:latest as builder

WORKDIR /app

# Copiar archivos de configuración
COPY package*.json ./
COPY tailwind.config.cjs ./
COPY tsconfig.json ./
COPY astro.config.mjs ./
COPY components.json ./

# Instalar dependencias
RUN npm install

# Copiar código fuente
COPY src ./src
COPY public ./public

# Construir la aplicación
RUN npm run build

# Etapa de producción
FROM node:latest
RUN npm install -g serve

WORKDIR /app
COPY --from=builder /app/dist ./dist

EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]