#!/usr/bin/env python3
"""
Kelly Audio2Face-3D Client
Specialized client for Kelly avatar facial animation
"""

import argparse
import asyncio
import sys
from pathlib import Path
import logging

# Add Audio2Face-3D client to path
sys.path.append(str(Path(__file__).parent.parent / "Audio2Face-3D-Samples" / "scripts" / "audio2face_3d_api_client"))

try:
    import a2f_3d.client.auth
    import a2f_3d.client.service
    from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
except ImportError as e:
    print(f"❌ Failed to import Audio2Face-3D modules: {e}")
    print("💡 Make sure you've installed the requirements:")
    print("   pip install -r Audio2Face-3D-Samples/scripts/audio2face_3d_api_client/requirements")
    print("   pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl")
    sys.exit(1)

def setup_logging():
    """Setup logging for Kelly client"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('kelly_audio2face/logs/kelly_client.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('KellyAudio2Face')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kelly Audio2Face-3D Client - Generate facial animation for Kelly avatar",
        epilog="NVIDIA CORPORATION. All rights reserved."
    )
    parser.add_argument("audio_file", help="Kelly audio file (WAV format, 16-bit PCM)")
    parser.add_argument("--config", default="config/kelly_config.yml", 
                       help="Kelly configuration file")
    parser.add_argument("--apikey", required=True, help="NVIDIA API Key")
    parser.add_argument("--function-id", required=True, help="Audio2Face-3D Function ID")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="claire", choices=["claire", "mark", "james"],
                       help="Audio2Face-3D model to use")
    return parser.parse_args()

async def process_kelly_audio(args, logger):
    """Process Kelly audio with Audio2Face-3D"""
    logger.info(f"🎬 Processing Kelly audio: {args.audio_file}")
    
    # Validate input files
    audio_path = Path(args.audio_file)
    config_path = Path(args.config)
    
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return False
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Setup gRPC connection
        metadata_args = [
            ("function-id", args.function_id),
            ("authorization", "Bearer " + args.apikey)
        ]
        
        logger.info("🔗 Connecting to Audio2Face-3D service...")
        channel = a2f_3d.client.auth.create_channel(
            uri="grpc.nvcf.nvidia.com:443", 
            use_ssl=True, 
            metadata=metadata_args
        )
        
        stub = A2FControllerServiceStub(channel)
        
        # Process audio stream
        logger.info("🎵 Processing audio stream...")
        stream = stub.ProcessAudioStream()
        
        write_task = asyncio.create_task(
            a2f_3d.client.service.write_to_stream(stream, str(config_path), str(audio_path))
        )
        read_task = asyncio.create_task(
            a2f_3d.client.service.read_from_stream(stream)
        )
        
        await write_task
        await read_task
        
        logger.info("✅ Kelly facial animation generated successfully!")
        logger.info(f"📁 Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing Kelly audio: {e}")
        return False

async def main():
    """Main function"""
    logger = setup_logging()
    logger.info("🚀 Starting Kelly Audio2Face-3D Client")
    
    args = parse_args()
    
    # Change to Kelly Audio2Face directory
    kelly_dir = Path("kelly_audio2face")
    if kelly_dir.exists():
        os.chdir(kelly_dir)
        logger.info(f"📁 Working in directory: {kelly_dir.absolute()}")
    
    success = await process_kelly_audio(args, logger)
    
    if success:
        logger.info("🎉 Kelly Audio2Face-3D processing completed successfully!")
        print("\n📋 Next steps:")
        print("1. Check output directory for blendshape data")
        print("2. Import animation data into your 3D software")
        print("3. Apply to Kelly's facial rig")
    else:
        logger.error("❌ Kelly Audio2Face-3D processing failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
