import { defineConfig } from 'astro/config';
import image from '@astrojs/image';
import purgecss from 'vite-plugin-purgecss';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  site: 'https://www.curiouskelly.com',
  output: 'static',
  trailingSlash: 'always',
  server: {
    port: 4321
  },
  envPrefix: ['PUBLIC_', 'CRM_', 'TURNSTILE_', 'RECAPTCHA_', 'LOCALE_'],
  integrations: [
    image({
      serviceEntryPoint: '@astrojs/image/sharp'
    })
  ],
  vite: {
    resolve: {
      alias: {
        '@components': path.resolve(__dirname, './src/components'),
        '@layouts': path.resolve(__dirname, './src/layouts'),
        '@lib': path.resolve(__dirname, './src/lib'),
        '@styles': path.resolve(__dirname, './styles')
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
          standard: [
            /^modal/,
            /^tooltip/,
            /^popover/,
            /^carousel/,
            /^slick/,
            /^iti/,
            /^fade/,
            /^show/,
            /^collapse/
          ]
        }
      })
    ]
  }
});




