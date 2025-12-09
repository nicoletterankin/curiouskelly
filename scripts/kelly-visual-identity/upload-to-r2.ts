/**
 * Kelly Visual Identity Pipeline - R2 Upload Script
 * 
 * Uploads generated Kelly poses to Cloudflare R2 bucket
 * and updates Supabase with asset metadata.
 * 
 * Usage: tsx scripts/kelly-visual-identity/upload-to-r2.ts [source-directory]
 */

import { S3Client, PutObjectCommand, HeadObjectCommand } from "@aws-sdk/client-s3";
import { createClient } from "@supabase/supabase-js";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

// Initialize S3 client for R2
const s3 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.CLOUDFLARE_R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY!,
  },
});

// Initialize Supabase client
const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface UploadResult {
  filename: string;
  r2Key: string;
  cdnUrl: string;
  success: boolean;
  error?: string;
}

/**
 * Calculate file hash for deduplication
 */
function calculateHash(buffer: Buffer): string {
  return crypto.createHash("sha256").update(buffer).digest("hex").substring(0, 16);
}

/**
 * Parse pose name from filename
 * Example: kelly_idle_v1.png → { pose: "idle", version: 1 }
 */
function parseFilename(filename: string): { pose: string; version: number } | null {
  const match = filename.match(/kelly_([a-z_]+)_v(\d+)\.(png|jpg|jpeg)/);
  if (!match) return null;
  
  return {
    pose: match[1],
    version: parseInt(match[2], 10)
  };
}

/**
 * Upload file to R2
 */
async function uploadToR2(
  filePath: string,
  r2Key: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const fileBuffer = fs.readFileSync(filePath);
    const contentType = filePath.endsWith(".png") ? "image/png" : "image/jpeg";
    
    await s3.send(new PutObjectCommand({
      Bucket: process.env.KELLY_ASSETS_BUCKET!,
      Key: r2Key,
      Body: fileBuffer,
      ContentType: contentType,
      CacheControl: "public, max-age=31536000, immutable",
      Metadata: {
        "uploaded-at": new Date().toISOString(),
        "file-hash": calculateHash(fileBuffer)
      }
    }));
    
    return { success: true };
  } catch (error: any) {
    return { success: false, error: error.message };
  }
}

/**
 * Check if file exists in R2
 */
async function fileExistsInR2(r2Key: string): Promise<boolean> {
  try {
    await s3.send(new HeadObjectCommand({
      Bucket: process.env.KELLY_ASSETS_BUCKET!,
      Key: r2Key
    }));
    return true;
  } catch {
    return false;
  }
}

/**
 * Insert asset metadata into Supabase
 */
async function insertAssetMetadata(
  filename: string,
  r2Key: string,
  pose: string,
  version: number
): Promise<{ success: boolean; error?: string }> {
  try {
    const { error } = await supabase
      .from("kelly_assets")
      .insert({
        filename,
        r2_key: r2Key,
        r2_bucket: process.env.KELLY_ASSETS_BUCKET!,
        pose_type: pose,
        status: "review", // Start in review status
        version,
        generation_model: "google-imagen-3",
        created_at: new Date().toISOString()
      });
    
    if (error) throw error;
    return { success: true };
  } catch (error: any) {
    return { success: false, error: error.message };
  }
}

/**
 * Upload all files from a directory
 */
async function uploadDirectory(sourceDir: string): Promise<UploadResult[]> {
  console.log("📤 Kelly Visual Identity Pipeline - R2 Upload");
  console.log("=".repeat(60));
  console.log(`📁 Source: ${sourceDir}`);
  console.log(`☁️  Bucket: ${process.env.KELLY_ASSETS_BUCKET}`);
  console.log(`🌐 CDN: ${process.env.KELLY_ASSETS_CDN_URL}`);
  console.log("");
  
  const results: UploadResult[] = [];
  
  // Get all image files
  const files = fs.readdirSync(sourceDir)
    .filter(f => /\.(png|jpg|jpeg)$/i.test(f))
    .filter(f => f.startsWith("kelly_"));
  
  console.log(`Found ${files.length} files to upload\n`);
  
  for (const filename of files) {
    const filePath = path.join(sourceDir, filename);
    const parsed = parseFilename(filename);
    
    if (!parsed) {
      console.log(`⚠️  Skipping ${filename} - invalid format`);
      results.push({
        filename,
        r2Key: "",
        cdnUrl: "",
        success: false,
        error: "Invalid filename format"
      });
      continue;
    }
    
    const { pose, version } = parsed;
    const r2Key = `staging/poses/${pose}/${filename}`;
    const cdnUrl = `${process.env.KELLY_ASSETS_CDN_URL}/${r2Key}`;
    
    console.log(`📤 Uploading: ${filename}`);
    console.log(`   Pose: ${pose}, Version: ${version}`);
    
    // Check if already exists
    const exists = await fileExistsInR2(r2Key);
    if (exists) {
      console.log(`   ⚠️  Already exists in R2, skipping upload`);
    } else {
      // Upload to R2
      const uploadResult = await uploadToR2(filePath, r2Key);
      if (!uploadResult.success) {
        console.log(`   ❌ Upload failed: ${uploadResult.error}`);
        results.push({
          filename,
          r2Key,
          cdnUrl,
          success: false,
          error: uploadResult.error
        });
        continue;
      }
      console.log(`   ✅ Uploaded to R2`);
    }
    
    // Insert metadata into Supabase
    const dbResult = await insertAssetMetadata(filename, r2Key, pose, version);
    if (!dbResult.success) {
      console.log(`   ⚠️  Database insert failed: ${dbResult.error}`);
    } else {
      console.log(`   ✅ Metadata saved to Supabase`);
    }
    
    console.log(`   🌐 URL: ${cdnUrl}`);
    console.log("");
    
    results.push({
      filename,
      r2Key,
      cdnUrl,
      success: true
    });
  }
  
  return results;
}

/**
 * Main execution
 */
async function main() {
  const sourceDir = process.argv[2] || path.join(process.cwd(), "generated-poses");
  
  if (!fs.existsSync(sourceDir)) {
    console.error(`❌ Source directory not found: ${sourceDir}`);
    process.exit(1);
  }
  
  // Verify environment variables
  const required = [
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_R2_ACCESS_KEY_ID",
    "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
    "KELLY_ASSETS_BUCKET",
    "KELLY_ASSETS_CDN_URL",
    "PUBLIC_SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY"
  ];
  
  const missing = required.filter(key => !process.env[key]);
  if (missing.length > 0) {
    console.error(`❌ Missing environment variables: ${missing.join(", ")}`);
    console.error("   See scripts/kelly-visual-identity/env-template.txt");
    process.exit(1);
  }
  
  const results = await uploadDirectory(sourceDir);
  
  // Summary
  console.log("=".repeat(60));
  console.log("📊 UPLOAD SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  
  console.log(`✅ Successful: ${successful}/${results.length}`);
  console.log(`❌ Failed: ${failed}/${results.length}`);
  
  if (failed > 0) {
    console.log("\n❌ Failed uploads:");
    results.filter(r => !r.success).forEach(r => {
      console.log(`   - ${r.filename}: ${r.error}`);
    });
  }
  
  console.log("\n🎯 NEXT STEPS:");
  console.log("1. Review uploaded images in Cloudflare R2 dashboard");
  console.log("2. Check asset metadata in Supabase kelly_assets table");
  console.log("3. Review and approve assets (update status to 'approved')");
  console.log("4. Move approved assets from staging/ to production/");
  console.log("5. Set is_hero=true for the best version of each pose");
}

if (require.main === module) {
  main().catch(console.error);
}

export { uploadDirectory, uploadToR2, insertAssetMetadata };








