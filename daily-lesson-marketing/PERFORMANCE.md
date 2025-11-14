# Performance Guide

## Performance Budgets

### Core Web Vitals Targets

- **LCP (Largest Contentful Paint)**: < 2.5s
- **CLS (Cumulative Layout Shift)**: < 0.1
- **INP (Interaction to Next Paint)**: < 200ms

### Resource Budgets

- **Homepage JS**: ≤ 80KB min+gzip
- **Total JS**: ≤ 200KB min+gzip
- **Total CSS**: ≤ 50KB min+gzip
- **Images**: Optimized WebP/AVIF with responsive srcset

## Optimization Strategies

### 1. JavaScript Optimization

- **Code Splitting**: Manual chunks for vendor libraries
- **Tree Shaking**: Unused code eliminated
- **Partial Hydration**: Only interactive components hydrated
- **Defer Non-Critical**: Scripts loaded asynchronously

### 2. CSS Optimization

- **Critical CSS**: Inline above-the-fold styles
- **PurgeCSS**: Unused styles removed
- **SCSS Modules**: Component-scoped styles

### 3. Image Optimization

- **Astro Image**: Automatic WebP/AVIF conversion
- **Responsive Images**: srcset for different viewports
- **Lazy Loading**: Below-the-fold images deferred
- **Aspect Ratio**: Prevents layout shift

### 4. Asset Loading

- **Preload**: Critical fonts and scripts
- **Preconnect**: Third-party domains
- **DNS Prefetch**: External resources

## Measuring Performance

### Lighthouse CI

Run locally:
```bash
npm run test:lighthouse
```

CI integration:
- Runs on every PR
- Fails if budgets exceeded
- Reports uploaded to temporary storage

### Real User Monitoring (RUM)

Optional endpoint `/api/rum` collects:
- LCP, CLS, INP metrics
- User agent and viewport
- Page URL

**Note**: Disabled in production by default. Enable via `ENABLE_RUM=true`.

## Performance Checklist

- [ ] LCP < 2.5s on mobile (Moto G)
- [ ] CLS < 0.1
- [ ] INP < 200ms
- [ ] Homepage JS ≤ 80KB gzipped
- [ ] Images use WebP/AVIF
- [ ] Critical CSS inlined
- [ ] Non-critical scripts deferred
- [ ] Fonts preloaded
- [ ] Third-party scripts loaded after consent

## Troubleshooting Performance Issues

### Slow LCP

- Optimize hero image (compress, use WebP)
- Inline critical CSS
- Preload hero image
- Reduce server response time

### High CLS

- Set explicit image dimensions
- Reserve space for ads/widgets
- Avoid dynamic content insertion above fold
- Use aspect-ratio CSS

### Slow INP

- Reduce JavaScript execution time
- Optimize event handlers
- Use requestIdleCallback for non-critical work
- Defer third-party scripts

### Large Bundle Size

- Audit bundle: `npm run build && npx vite-bundle-visualizer`
- Split vendor chunks
- Remove unused dependencies
- Use dynamic imports for heavy libraries

## Performance Monitoring

### CI/CD Integration

Lighthouse CI runs automatically:
- On every PR
- On main branch deployments
- Reports available in GitHub Actions

### Production Monitoring

Consider integrating:
- Google Analytics Core Web Vitals
- New Relic Browser
- Datadog RUM
- Custom RUM endpoint (if enabled)

## Best Practices

1. **Measure First**: Always measure before optimizing
2. **Set Budgets**: Enforce budgets in CI
3. **Monitor Trends**: Track performance over time
4. **Test Real Devices**: Emulators don't catch all issues
5. **Optimize Third-Party**: Load marketing scripts after consent











