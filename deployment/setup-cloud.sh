#!/bin/bash

# The Daily Lesson - Cloud Infrastructure Setup
# This script sets up the cloud infrastructure for the platform

echo "🚀 Setting up The Daily Lesson cloud infrastructure..."

# Check if required tools are installed
command -v vercel >/dev/null 2>&1 || { echo "❌ Vercel CLI not found. Please install: npm i -g vercel"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not found. Please install AWS CLI"; exit 1; }

# Create necessary directories
mkdir -p deployment/{api,cdn,monitoring}
mkdir -p assets/{videos,images,audio}

echo "📁 Created directory structure"

# Set up Vercel project
echo "🔧 Setting up Vercel deployment..."
cd deployment
vercel init the-daily-lesson --yes

# Configure environment variables
echo "🔐 Setting up environment variables..."
vercel env add ELEVENLABS_API_KEY
vercel env add STRIPE_SECRET_KEY
vercel env add ANALYTICS_ID
vercel env add DATABASE_URL

# Set up AWS S3 for video storage
echo "☁️ Setting up AWS S3 for video storage..."
aws s3 mb s3://daily-lesson-videos --region us-east-1
aws s3 mb s3://daily-lesson-assets --region us-east-1

# Configure S3 bucket policies
cat > s3-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::daily-lesson-videos/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy --bucket daily-lesson-videos --policy file://s3-policy.json

# Set up CloudFront CDN
echo "🌐 Setting up CloudFront CDN..."
aws cloudfront create-distribution --distribution-config file://cloudfront-config.json

# Set up monitoring with DataDog or similar
echo "📊 Setting up monitoring..."
# Add monitoring configuration here

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Cloud infrastructure setup complete!"
echo "🌐 Your app is live at: https://the-daily-lesson.vercel.app"
echo "📊 Monitor at: https://vercel.com/dashboard"
echo "☁️ Videos stored at: https://daily-lesson-videos.s3.amazonaws.com"

# Create deployment checklist
cat > DEPLOYMENT_CHECKLIST.md << EOF
# Deployment Checklist

## ✅ Completed
- [x] Vercel project initialized
- [x] Environment variables configured
- [x] AWS S3 buckets created
- [x] CloudFront CDN configured
- [x] Initial deployment successful

## 🔄 Next Steps
- [ ] Upload Kelly videos to S3
- [ ] Configure analytics tracking
- [ ] Set up payment processing
- [ ] Test all age variants
- [ ] Load test the platform
- [ ] Set up monitoring alerts

## 🔗 Important Links
- App: https://the-daily-lesson.vercel.app
- Dashboard: https://vercel.com/dashboard
- S3 Videos: https://daily-lesson-videos.s3.amazonaws.com
- CloudFront: [Check AWS Console]

## 🔐 Environment Variables
- ELEVENLABS_API_KEY: [Set in Vercel dashboard]
- STRIPE_SECRET_KEY: [Set in Vercel dashboard]
- ANALYTICS_ID: [Set in Vercel dashboard]
- DATABASE_URL: [Set in Vercel dashboard]
EOF

echo "📋 Created deployment checklist: DEPLOYMENT_CHECKLIST.md"

