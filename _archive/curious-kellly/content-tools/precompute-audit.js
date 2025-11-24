#!/usr/bin/env node

/**
 * Pre-computation Audit Script
 * Checks multilingual completeness (EN/ES/FR) for all lessons
 */

const fs = require('fs');
const path = require('path');

const WORKSPACE_ROOT = path.resolve(__dirname, '../..');
const LESSONS_DIR = path.join(WORKSPACE_ROOT, 'curious-kellly', 'backend', 'config', 'lessons');
const REQUIRED_AGE_GROUPS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
const REQUIRED_LANGUAGES = ['en', 'es', 'fr'];
const REQUIRED_SECTIONS = ['welcome', 'mainContent', 'keyPoints', 'interactionPrompts', 'wisdomMoment', 'cta', 'summary'];

/**
 * Audit multilingual completeness for a single lesson
 */
function auditLesson(lessonPath) {
  const lessonData = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  const audit = {
    filename: path.basename(lessonPath),
    id: lessonData.id || lessonData.lesson_id,
    ageGroups: {},
    overall: {
      hasEn: false,
      hasEs: false,
      hasFr: false,
      completeness: {
        en: 0,
        es: 0,
        fr: 0
      },
      missingSections: []
    }
  };
  
  if (!lessonData.ageVariants) {
    audit.error = 'No ageVariants found';
    return audit;
  }
  
  let totalSections = 0;
  let enSections = 0;
  let esSections = 0;
  let frSections = 0;
  
  REQUIRED_AGE_GROUPS.forEach(ageGroup => {
    const variant = lessonData.ageVariants[ageGroup];
    if (!variant) {
      audit.ageGroups[ageGroup] = {
        status: 'missing',
        languages: {}
      };
      return;
    }
    
    const ageGroupAudit = {
      status: 'present',
      languages: {
        en: { present: false, sections: {}, missingSections: [] },
        es: { present: false, sections: {}, missingSections: [] },
        fr: { present: false, sections: {}, missingSections: [] }
      }
    };
    
    if (variant.language) {
      REQUIRED_LANGUAGES.forEach(lang => {
        const langData = variant.language[lang];
        if (langData) {
          ageGroupAudit.languages[lang].present = true;
          audit.overall[`has${lang.toUpperCase()}`] = true;
          
          let sectionCount = 0;
          REQUIRED_SECTIONS.forEach(section => {
            totalSections++;
            if (langData[section]) {
              ageGroupAudit.languages[lang].sections[section] = true;
              sectionCount++;
              
              if (lang === 'en') enSections++;
              if (lang === 'es') esSections++;
              if (lang === 'fr') frSections++;
            } else {
              ageGroupAudit.languages[lang].sections[section] = false;
              ageGroupAudit.languages[lang].missingSections.push(section);
              audit.overall.missingSections.push(`${ageGroup}.${lang}.${section}`);
            }
          });
        } else {
          REQUIRED_SECTIONS.forEach(section => {
            totalSections++;
            ageGroupAudit.languages[lang].missingSections.push(section);
            audit.overall.missingSections.push(`${ageGroup}.${lang}.${section}`);
          });
        }
      });
    } else {
      // No language structure at all
      REQUIRED_LANGUAGES.forEach(lang => {
        REQUIRED_SECTIONS.forEach(section => {
          totalSections++;
          audit.overall.missingSections.push(`${ageGroup}.${lang}.${section}`);
        });
      });
    }
    
    audit.ageGroups[ageGroup] = ageGroupAudit;
  });
  
  // Calculate completeness percentages
  if (totalSections > 0) {
    audit.overall.completeness.en = (enSections / (REQUIRED_AGE_GROUPS.length * REQUIRED_SECTIONS.length) * 100).toFixed(1);
    audit.overall.completeness.es = (esSections / (REQUIRED_AGE_GROUPS.length * REQUIRED_SECTIONS.length) * 100).toFixed(1);
    audit.overall.completeness.fr = (frSections / (REQUIRED_AGE_GROUPS.length * REQUIRED_SECTIONS.length) * 100).toFixed(1);
  }
  
  return audit;
}

/**
 * Audit all lessons
 */
function auditAllLessons() {
  const results = {
    timestamp: new Date().toISOString(),
    total: 0,
    complete: 0,
    partial: 0,
    missing: 0,
    lessons: []
  };
  
  if (!fs.existsSync(LESSONS_DIR)) {
    console.error(`Error: Lessons directory not found: ${LESSONS_DIR}`);
    return results;
  }
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => 
    f.endsWith('.json') && !f.startsWith('.')
  );
  
  results.total = files.length;
  
  console.log(`\n🔍 Auditing multilingual completeness for ${files.length} lessons...\n`);
  
  files.forEach((file, index) => {
    const filePath = path.join(LESSONS_DIR, file);
    console.log(`[${index + 1}/${files.length}] ${file}`);
    
    try {
      const audit = auditLesson(filePath);
      results.lessons.push(audit);
      
      // Categorize completeness
      const hasAll = audit.overall.hasEn && audit.overall.hasEs && audit.overall.hasFr;
      const hasEn = audit.overall.hasEn;
      
      if (hasAll) {
        results.complete++;
      } else if (hasEn) {
        results.partial++;
      } else {
        results.missing++;
      }
    } catch (e) {
      results.lessons.push({
        filename: file,
        error: e.message
      });
      results.missing++;
    }
  });
  
  return results;
}

/**
 * Generate precomputation report
 */
function generatePrecomputationReport(results) {
  const report = {
    ...results,
    summary: {
      total: results.total,
      complete: results.complete,
      partial: results.partial,
      missing: results.missing,
      completenessRate: results.total > 0 ? (results.complete / results.total * 100).toFixed(1) + '%' : '0%'
    },
    byCompleteness: {
      complete: results.lessons.filter(l => 
        l.overall && l.overall.hasEn && l.overall.hasEs && l.overall.hasFr
      ),
      partial: results.lessons.filter(l => 
        l.overall && l.overall.hasEn && (!l.overall.hasEs || !l.overall.hasFr)
      ),
      missing: results.lessons.filter(l => 
        !l.overall || !l.overall.hasEn
      )
    },
    languageStats: {
      en: { present: 0, missing: 0 },
      es: { present: 0, missing: 0 },
      fr: { present: 0, missing: 0 }
    },
    missingSectionsBreakdown: {}
  };
  
  // Count language presence
  results.lessons.forEach(lesson => {
    if (lesson.overall) {
      ['en', 'es', 'fr'].forEach(lang => {
        if (lesson.overall[`has${lang.toUpperCase()}`]) {
          report.languageStats[lang].present++;
        } else {
          report.languageStats[lang].missing++;
        }
      });
      
      // Analyze missing sections
      lesson.overall.missingSections.forEach(section => {
        const parts = section.split('.');
        const lang = parts[1];
        const sectionName = parts[2];
        const key = `${lang}.${sectionName}`;
        report.missingSectionsBreakdown[key] = (report.missingSectionsBreakdown[key] || 0) + 1;
      });
    }
  });
  
  return report;
}

// CLI
if (require.main === module) {
  const results = auditAllLessons();
  const report = generatePrecomputationReport(results);
  
  console.log('\n' + '='.repeat(60));
  console.log('🌍 MULTILINGUAL PRE-COMPUTATION AUDIT');
  console.log('='.repeat(60));
  console.log(`Total lessons: ${report.summary.total}`);
  console.log(`✅ Complete (EN+ES+FR): ${report.summary.complete}`);
  console.log(`⚠️  Partial (EN only): ${report.summary.partial}`);
  console.log(`❌ Missing: ${report.summary.missing}`);
  console.log(`📈 Completeness rate: ${report.summary.completenessRate}`);
  
  console.log('\n📊 Language Presence:');
  ['en', 'es', 'fr'].forEach(lang => {
    const stats = report.languageStats[lang];
    const presentRate = report.summary.total > 0 
      ? (stats.present / report.summary.total * 100).toFixed(1) + '%'
      : '0%';
    console.log(`  ${lang.toUpperCase()}: ${stats.present}/${report.summary.total} (${presentRate})`);
  });
  
  if (Object.keys(report.missingSectionsBreakdown).length > 0) {
    console.log('\n📋 Missing Sections Breakdown (top 10):');
    Object.entries(report.missingSectionsBreakdown)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([section, count]) => {
        console.log(`  ${section}: ${count} missing`);
      });
  }
  
  if (report.byCompleteness.partial.length > 0) {
    console.log('\n⚠️  Lessons Missing Translations:');
    report.byCompleteness.partial.forEach(lesson => {
      const missing = [];
      if (!lesson.overall.hasEs) missing.push('ES');
      if (!lesson.overall.hasFr) missing.push('FR');
      console.log(`  ${lesson.filename}: Missing ${missing.join(', ')}`);
      console.log(`    EN: ${lesson.overall.completeness.en}% | ES: ${lesson.overall.completeness.es}% | FR: ${lesson.overall.completeness.fr}%`);
    });
  }
  
  if (report.byCompleteness.missing.length > 0) {
    console.log('\n❌ Lessons Missing All Content:');
    report.byCompleteness.missing.forEach(lesson => {
      console.log(`  ${lesson.filename}: ${lesson.error || 'No language structure'}`);
    });
  }
  
  // Save report
  const reportPath = path.join(WORKSPACE_ROOT, 'docs', 'phasedna', 'precomputation-audit.json');
  const reportDir = path.dirname(reportPath);
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n💾 Precomputation audit saved to: ${reportPath}\n`);
}

module.exports = {
  auditLesson,
  auditAllLessons,
  generatePrecomputationReport
};

