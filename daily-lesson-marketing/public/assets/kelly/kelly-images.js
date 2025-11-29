/**
 * Kelly Image Helper
 * Auto-generated - do not edit manually
 * Generated: 2025-11-29T19:34:35.083Z
 */

export const kellyImages = {
  hero: {
    srcset: {
      webp: `
        /assets/kelly/production/hero/kelly-hero-mobile.webp 640w,
        /assets/kelly/production/hero/kelly-hero-tablet.webp 1280w,
        /assets/kelly/production/hero/kelly-hero-desktop.webp 1920w,
        /assets/kelly/production/hero/kelly-hero-4k.webp 3840w
      `.trim(),
      jpeg: `
        /assets/kelly/production/hero/kelly-hero-mobile.jpg 640w,
        /assets/kelly/production/hero/kelly-hero-tablet.jpg 1280w,
        /assets/kelly/production/hero/kelly-hero-desktop.jpg 1920w,
        /assets/kelly/production/hero/kelly-hero-4k.jpg 3840w
      `.trim()
    },
    fallback: '/assets/kelly/production/hero/kelly-hero-desktop.jpg'
  },
  
  getAvatar(expression = 'curious', size = 256, format = 'webp') {
    const validExpressions = ["curious","explaining","celebrating","listening","wisdom"];
    const validSizes = [512,256,128,64];
    
    const exp = validExpressions.includes(expression) ? expression : 'curious';
    const sz = validSizes.includes(size) ? size : 256;
    const fmt = format === 'jpeg' ? 'jpg' : 'webp';
    
    return `/assets/kelly/production/avatars/${exp}/kelly-${exp}-${sz}.${fmt}`;
  },
  
  social: {
    og: '/assets/kelly/production/social/og-image.jpg',
    twitter: '/assets/kelly/production/social/twitter-card.jpg',
    linkedin: '/assets/kelly/production/social/linkedin-banner.jpg',
    instagram: '/assets/kelly/production/social/instagram-square.jpg',
    instagramStory: '/assets/kelly/production/social/instagram-story.jpg'
  }
};

export default kellyImages;
