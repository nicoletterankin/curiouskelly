
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

function getArgValue(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  if (idx === -1) return undefined;
  return process.argv[idx + 1];
}

function getIntArg(flag: string, fallback: number): number {
  const raw = getArgValue(flag);
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return fallback;
  return n;
}

const START_DAY = getIntArg('--start', 2);
const END_DAY = getIntArg('--end', 30);
const DELAY_MS = 30000; // 30 seconds delay

async function runBatch() {
  if (!Number.isFinite(START_DAY) || !Number.isFinite(END_DAY) || START_DAY < 1 || END_DAY < START_DAY) {
    console.error('❌ Invalid range. Usage: npx tsx scripts/batch-visual-generation.ts --start <n> --end <n>');
    process.exit(1);
  }

  console.log(`🚀 Starting batch visual generation for Days ${START_DAY}-${END_DAY}`);
  console.log(`⏱️  Delay between days: ${DELAY_MS}ms`);

  for (let day = START_DAY; day <= END_DAY; day++) {
    console.log(`\n📸 Processing Day ${day}...`);
    
    try {
      const { stdout, stderr } = await execAsync(`npx tsx scripts/generate-lesson-visuals.ts --day ${day}`);
      console.log(stdout);
      
      if (stderr && !stderr.includes('Debugger attached')) {
        console.error(`⚠️  Stderr for Day ${day}:`, stderr);
      }
      
      // Check if we hit a rate limit error in the output
      if (stdout.includes('429') || stdout.includes('RESOURCE_EXHAUSTED') || stdout.includes('Quota exceeded')) {
        console.error(`🛑 Rate limit detected on Day ${day}. Stopping batch.`);
        break;
      }
      
      if (stdout.includes('✅ Day')) {
        console.log(`✅ Day ${day} successfully processed.`);
      } else {
        console.warn(`⚠️  Day ${day} might have had issues.`);
      }

    } catch (error: any) {
      console.error(`❌ Fatal error executing Day ${day}:`, error.message);
      if (error.stdout && (error.stdout.includes('429') || error.stdout.includes('RESOURCE_EXHAUSTED'))) {
         console.error(`🛑 Rate limit detected (in catch). Stopping batch.`);
         break;
      }
    }

    if (day < END_DAY) {
      console.log(`⏳ Waiting ${DELAY_MS/1000} seconds...`);
      await sleep(DELAY_MS);
    }
  }

  console.log('\n✨ Batch run finished.');
}

runBatch();
