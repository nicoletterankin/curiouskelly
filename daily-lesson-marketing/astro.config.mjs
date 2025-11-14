import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';
import purgecss from 'vite-plugin-purgecss';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  site: 'https://www.thedailylesson.com',
  trailingSlash: 'always',
  server: {
    port: 4321
  },
  envPrefix: ['PUBLIC_', 'CRM_', 'TURNSTILE_', 'RECAPTCHA_', 'LOCALE_'],
  integrations: [],
  adapter: process.env.NODE_ENV === 'development' ? undefined : vercel(),
  output: 'static',
  vite: {
    resolve: {
      alias: {
        '@components': path.resolve(__dirname, './src/components'),
        '@layouts': path.resolve(__dirname, './src/layouts'),
        '@lib': path.resolve(__dirname, './src/lib'),
        '@styles': path.resolve(__dirname, './src/styles'),
        '@': path.resolve(__dirname, './src')
      }
    },
    plugins: [
      purgecss({
        content: [
          './src/**/*.astro',
          './src/**/*.ts',
          './src/**/*.tsx',
          './src/**/*.js',
          './styles/**/*.scss'
        ],
        safelist: {
          standard: [/^modal/, /^tooltip/, /^popover/, /^carousel/, /^slick/, /^iti/, /^fade/, /^show/, /^collapse/]
        }
      })
    ],
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'jquery-vendor': ['jquery', 'jquery-migrate'],
            'bootstrap-vendor': ['bootstrap', 'popper.js'],
            'carousel-vendor': ['slick-carousel'],
            'intl-vendor': ['intl-tel-input']
          }
        }
      }
    },
    optimizeDeps: {
      include: ['jquery', 'bootstrap', 'slick-carousel', 'intl-tel-input']
    },
    css: {
      preprocessorOptions: {
        scss: {
          includePaths: [path.resolve(__dirname, './node_modules')]
        }
      }
    }
  },
  build: {
    inlineStylesheets: 'auto',
    assets: '_assets'
  }
});

