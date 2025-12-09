
import * as fs from "fs";
import * as path from "path";

const OUTPUT_BASE_DIR = path.join(process.cwd(), "public", "kelly", "phases");
const phases = ['hook', 'q1', 'q2', 'q3', 'wisdom'];

console.log("Checking for missing infographics...");

const missingDays = [];

for (let i = 1; i <= 365; i++) {
    const dayPadded = String(i).padStart(3, '0');
    const dayDir = path.join(OUTPUT_BASE_DIR, dayPadded);
    
    if (!fs.existsSync(dayDir)) {
        missingDays.push(i);
        continue;
    }

    let isMissing = false;
    for (const p of phases) {
        if (!fs.existsSync(path.join(dayDir, `${p}.png`))) {
            isMissing = true;
            break;
        }
    }

    if (isMissing) {
        missingDays.push(i);
    }
}

console.log(`Found ${missingDays.length} days with missing infographics.`);
console.log(JSON.stringify(missingDays));






