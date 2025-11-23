
const fs = require('fs');
const path = require('path');

const LESSONS_DIR = path.resolve(__dirname, '../backend/config/lessons');
const EXPECTED_AGE_GROUPS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
const EXPECTED_LANGUAGES = ['en', 'es', 'fr'];

function verifyLesson(filename) {
  const filePath = path.join(LESSONS_DIR, filename);
  try {
    if (!fs.existsSync(filePath)) {
        return { filename, status: 'MISSING' };
    }
    
    const content = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(content);
    
    const report = {
        filename,
        id: data.id,
        title: data.title,
        version: data.version,
        day: data.calendar?.day,
        status: 'OK',
        issues: []
    };

    // Check Schema Version
    if (data.version !== '2.0.0') {
        report.issues.push(`Version mismatch: ${data.version}`);
    }

    // Check Age Variants
    if (!data.ageVariants) {
        report.status = 'ERROR';
        report.issues.push('Missing ageVariants');
    } else {
        EXPECTED_AGE_GROUPS.forEach(age => {
            if (!data.ageVariants[age]) {
                report.issues.push(`Missing age variant: ${age}`);
            } else {
                // Check Languages
                if (!data.ageVariants[age].language) {
                    report.issues.push(`Missing language object for ${age}`);
                } else {
                    EXPECTED_LANGUAGES.forEach(lang => {
                        if (!data.ageVariants[age].language[lang]) {
                             report.issues.push(`Missing language: ${age}.${lang}`);
                        }
                    });
                }
            }
        });
    }

    return report;

  } catch (e) {
      return { filename, status: 'ERROR', issues: [e.message] };
  }
}

const files = fs.readdirSync(LESSONS_DIR).filter(f => f.endsWith('.json') && !f.startsWith('.'));
console.log(`Found ${files.length} files.`);

const results = files.map(verifyLesson);

// Sort by Day if possible
results.sort((a, b) => (a.day || 999) - (b.day || 999));

console.log(JSON.stringify(results, null, 2));


