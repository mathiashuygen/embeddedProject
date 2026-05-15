FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --production

# Copy application
COPY server.js .

# Create data directory and set permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Create a non-root user and switch to it
RUN addgroup -g 1001 -S nodejs && \
  adduser -S nextjs -u 1001 && \
  chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 8080

CMD ["node", "server.js"]
