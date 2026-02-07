/**
 * LIST ALL ZIGGURAT AUDIO SEGMENTS
 * Returns the full URLs for all 16 generated segments
 */

import { NextResponse } from 'next/server'
import { list } from '@vercel/blob'

const ZIGGURAT_SCRIPT = [
  { id: '01', text: "Let me show you something." },
  { id: '02', text: "In nineteen sixty-eight, architect William Pereira—the man who designed the Transamerica Pyramid—was commissioned to build something unprecedented." },
  { id: '03', text: "A facility for seven thousand engineers to manufacture the future." },
  { id: '04', text: "Ninety-two acres. One million square feet. Four miles from the Pacific." },
  { id: '05', text: "The company that commissioned it never moved in. The contract fell through before the doors ever opened." },
  { id: '06', text: "So the federal government used it. For paperwork. Fifty years of paperwork." },
  { id: '07', text: "That's not what this building was made for." },
  { id: '08', text: "Now they're selling it. Three auctions. No deal has closed. The last bidder planned to tear it down." },
  { id: '09', text: "Our founder sees something different." },
  { id: '10', text: "Eighteen-inch raised floors—perfect for data centers and manufacturing. Redundant power systems. Loading docks built for high-volume logistics." },
  { id: '11', text: "Every problem the government lists is exactly the infrastructure Lesson of the Day needs to deliver on SDG 4." },
  { id: '12', text: "This is where iLearn computers should be manufactured. This is where global curriculum should be printed. This is where The Daily Lesson broadcasts to the world." },
  { id: '13', text: "William Pereira designed this building to manufacture the future." },
  { id: '14', text: "We're going to finish what he started." },
  { id: '15', text: "Will you join us?" },
  { id: '16', text: "Lesson of the Day." },
]

export async function GET() {
  try {
    const blobs = await list({ prefix: 'video/ziggurat/' })
    
    const segments = ZIGGURAT_SCRIPT.map(seg => {
      const existing = blobs.blobs.find(b => b.pathname.includes(`segment_${seg.id}`))
      return {
        id: seg.id,
        text: seg.text,
        url: existing?.url || null,
        generated: !!existing,
        size: existing?.size || 0,
      }
    })

    const generatedCount = segments.filter(s => s.generated).length
    const allUrls = segments.filter(s => s.url).map(s => s.url)
    
    // Generate FFmpeg script with actual URLs
    const ffmpegScript = `# Ziggurat Video Assembly Script
# Run this after downloading all audio segments

# 1. Download all audio segments
${segments.filter(s => s.url).map(s => `curl -o segment_${s.id}.mp3 "${s.url}"`).join('\n')}

# 2. Create silence file (0.5 seconds)
ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 0.5 -q:a 9 -acodec libmp3lame silence.mp3

# 3. Create concat list
cat > concat.txt << 'EOF'
${segments.filter(s => s.url).map((s, i) => `file 'segment_${s.id}.mp3'\n${i < segments.length - 1 ? "file 'silence.mp3'" : ''}`).join('\n')}
EOF

# 4. Concatenate all audio
ffmpeg -f concat -safe 0 -i concat.txt -c copy combined_audio.mp3

# 5. Add audio to video with B&W filter
ffmpeg -i LAGUNA_RIDGE_FINAL_POLISHED.mp4 -i combined_audio.mp3 -map 0:v -map 1:a -c:v libx264 -vf "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3:0,eq=contrast=1.1:brightness=-0.05,noise=alls=10:allf=t" -c:a aac -shortest LAGUNA_RIDGE_FINAL_BW.mp4

echo "Done! Output: LAGUNA_RIDGE_FINAL_BW.mp4"
`

    // PowerShell version
    const powershellScript = `# Ziggurat Video Assembly Script (PowerShell)
# Run this after downloading all audio segments

# 1. Download all audio segments
${segments.filter(s => s.url).map(s => `Invoke-WebRequest -Uri "${s.url}" -OutFile "segment_${s.id}.mp3"`).join('\n')}

# 2. Create silence file (0.5 seconds)
ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 0.5 -q:a 9 -acodec libmp3lame silence.mp3

# 3. Create concat list
@"
${segments.filter(s => s.url).map((s, i) => `file 'segment_${s.id}.mp3'${i < segments.length - 1 ? "\nfile 'silence.mp3'" : ''}`).join('\n')}
"@ | Out-File -Encoding ascii concat.txt

# 4. Concatenate all audio
ffmpeg -f concat -safe 0 -i concat.txt -c copy combined_audio.mp3

# 5. Add audio to video with B&W filter
ffmpeg -i LAGUNA_RIDGE_FINAL_POLISHED.mp4 -i combined_audio.mp3 -map 0:v -map 1:a -c:v libx264 -vf "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3:0,eq=contrast=1.1:brightness=-0.05,noise=alls=10:allf=t" -c:a aac -shortest LAGUNA_RIDGE_FINAL_BW.mp4

Write-Host "Done! Output: LAGUNA_RIDGE_FINAL_BW.mp4"
`

    return NextResponse.json({
      status: generatedCount === 16 ? 'complete' : generatedCount > 0 ? 'partial' : 'pending',
      generated: generatedCount,
      total: 16,
      segments,
      allUrls,
      scripts: {
        bash: ffmpegScript,
        powershell: powershellScript,
      },
      // Direct copy-paste URLs
      copyPasteUrls: segments.filter(s => s.url).map(s => s.url).join('\n'),
    })
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}
