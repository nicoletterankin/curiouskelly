const fs = require('fs');
const path = require('path');

const LESSONS_DIR = path.join(__dirname, 'curious-kellly/backend/config/lessons');
const ARCHIVE_DIR = path.join(LESSONS_DIR, 'archive');

if (!fs.existsSync(ARCHIVE_DIR)) {
    fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
}

function migrateLesson(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const json = JSON.parse(content);
    let modified = false;

    // 1. Move Metadata to Root
    if (json.metadata) {
        ['category', 'difficulty', 'tags', 'prerequisites', 'learningOutcomes'].forEach(field => {
            if (json.metadata[field]) {
                json[field] = json.metadata[field];
                delete json.metadata[field];
                modified = true;
            }
        });
        // Keep metadata object if it still has props (like accessibility), otherwise maybe keep it for future
    }

    // 2. Ensure Version is 2.0.0
    if (json.version !== '2.0.0') {
        json.version = '2.0.0';
        modified = true;
    }

    // 3. Migrate Age Variants to Phases
    if (json.ageVariants) {
        Object.keys(json.ageVariants).forEach(age => {
            const variant = json.ageVariants[age];

            // If teachingMoments exist (old V1 format), migrate to phases and cleanup
            if (variant.teachingMoments) {
                console.log(`Migrating ${path.basename(filePath)} (${age}) to phases (overwrite)...`);
                
                const phases = [
                    {
                        id: 'welcome',
                        type: 'welcome',
                        duration: 30,
                        content: variant.language?.en?.welcome || "Welcome!",
                        expressionCues: []
                    },
                    {
                        id: 'teaching',
                        type: 'teaching',
                        duration: 300, // default 5 mins
                        content: variant.language?.en?.mainContent || "Main lesson content.",
                        teachingMoments: (variant.teachingMoments || []).map(tm => ({
                            concept: tm.type || "concept",
                            explanation: tm.content || "explanation",
                            ageAppropriate: "age-appropriate explanation"
                        })),
                        expressionCues: (variant.expressionCues || []).map(ec => ({
                            timestamp: ec.offset || 0,
                            type: ec.type || "neutral",
                            intensity: ec.intensity || "medium",
                            description: ec.gazeTarget ? `Look at ${ec.gazeTarget}` : "expression"
                        }))
                    },
                    {
                        id: 'wisdom',
                        type: 'wisdom',
                        duration: 60,
                        content: variant.language?.en?.wisdomMoment || "Wisdom moment.",
                        expressionCues: []
                    }
                ];

                variant.phases = phases;
                
                // Clean up old V1 fields
                delete variant.teachingMoments;
                delete variant.expressionCues;
                
                modified = true;
            }
        });
    }

    if (modified) {
        fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
        console.log(`✅ Migrated: ${path.basename(filePath)}`);
    } else {
        console.log(`Skipped (No changes): ${path.basename(filePath)}`);
    }
}

// Run
const files = fs.readdirSync(LESSONS_DIR).filter(f => f.endsWith('.json'));
files.forEach(f => {
    try {
        migrateLesson(path.join(LESSONS_DIR, f));
    } catch (e) {
        console.error(`Error migrating ${f}:`, e.message);
    }
});

