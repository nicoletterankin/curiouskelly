#!/bin/bash
# Copy generated audio files to Unity project
# Usage: ./copy_audio_to_unity_project.sh

set -e  # Exit on error

echo "============================================================"
echo "Copying Audio Files to Unity Project"
echo "============================================================"

# Paths
AUDIO_SOURCE="curious-kellly/backend/config/audio"
UNITY_DEST="digital-kelly/engines/kelly_unity_player/Assets/Kelly/Audio"

# Create Unity audio directory if it doesn't exist
mkdir -p "$UNITY_DEST"

# Copy all lesson audio folders
echo "Copying lesson audio files..."

lessons=(
    "the-sun"
    "puppies"
    "the-ocean"
    "the-moon"
    "water-cycle"
    "molecular-biology-dna"
    "creative-writing-dna"
    "poetry-dna"
    "dance-expression-dna"
    "negotiation-skills-dna"
)

total_files=0
total_size=0

for lesson in "${lessons[@]}"; do
    echo "  Copying $lesson..."

    # Create lesson directory in Unity
    mkdir -p "$UNITY_DEST/$lesson"

    # Copy MP3 files
    if [ -d "$AUDIO_SOURCE/$lesson" ]; then
        cp -v "$AUDIO_SOURCE/$lesson"/*.mp3 "$UNITY_DEST/$lesson/" 2>/dev/null || echo "    No files found for $lesson"

        # Count files and size
        file_count=$(find "$UNITY_DEST/$lesson" -name "*.mp3" | wc -l)
        dir_size=$(du -sm "$UNITY_DEST/$lesson" | cut -f1)

        total_files=$((total_files + file_count))
        total_size=$((total_size + dir_size))

        echo "    ✅ $file_count files ($dir_size MB)"
    else
        echo "    ⚠️  Directory not found: $AUDIO_SOURCE/$lesson"
    fi
done

echo ""
echo "============================================================"
echo "Copy Complete!"
echo "============================================================"
echo "Total files copied: $total_files"
echo "Total size: $total_size MB"
echo ""
echo "Next steps:"
echo "1. Open Unity project: digital-kelly/engines/kelly_unity_player/"
echo "2. Unity will import the audio files automatically"
echo "3. Check Assets/Kelly/Audio/ in Unity Project window"
echo "4. Test audio playback with LessonAudioPlayer.cs"
echo "============================================================"
