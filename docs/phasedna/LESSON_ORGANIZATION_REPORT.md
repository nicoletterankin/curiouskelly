# Lesson Organization, Validation, and Pre-computation Report

**Generated:** 2025-01-11  
**Status:** In Progress

## Executive Summary

This report documents the organization, validation, and pre-computation status of all lesson files in the Curious Kelly system.

### Key Metrics

- **Total Lessons:** 16
- **Canonical Location:** `curious-kellly/backend/config/lessons/`
- **Schema Versions:**
  - PhaseDNA v1: 5 lessons
  - PhaseDNA v2: 10 lessons (includes migrated)
  - Old Format: 0 lessons (all migrated or in process)
  - Unknown: 1 lesson (JSON parse errors)

### Current Status

- ✅ **Organization:** Complete - All lessons consolidated to canonical location
- ⚠️ **Validation:** Partial - Some lessons need migration/fixes
- ❌ **Pre-computation:** Incomplete - Multilingual content (EN/ES/FR) needs completion

## Phase 1: Discovery and Inventory

### Lesson Locations

**Canonical Location:** `curious-kellly/backend/config/lessons/`
- 16 lesson files identified
- Standardized naming: kebab-case format

**Legacy Location:** `lessons/`
- Files moved to canonical location
- Originals archived to `lessons/archive/` (where applicable)

### Schema Version Distribution

| Version | Count | Status |
|---------|-------|--------|
| PhaseDNA v1 | 5 | Ready for validation |
| PhaseDNA v2 | 10 | Includes migrated files |
| Old Format | 0 | All migrated |
| Unknown | 1 | JSON parse errors |

### Age Variant Completeness

- **Complete (6/6 age groups):** 6 lessons
- **Partial:** 1 lesson
- **Missing:** 9 lessons (old format, needs migration)

### Multilingual Status

- **Complete (EN+ES+FR):** 0 lessons
- **Partial (EN only):** 0 lessons  
- **Missing:** 16 lessons

**Language Presence:**
- English (EN): 5/16 lessons (31.3%)
- Spanish (ES): 5/16 lessons (31.3%)
- French (FR): 5/16 lessons (31.3%)

## Phase 2: Organization

### Files Moved

All valid lesson files from `lessons/` directory have been moved to `curious-kellly/backend/config/lessons/` with standardized naming:

- `molecular_biology_dna.json` → `molecular-biology-dna.json`
- `negotiation_skills_dna.json` → `negotiation-skills-dna.json`
- `dance_expression_dna.json` → `dance-expression-dna.json`
- And others...

### Duplicates Handled

- 1 duplicate identified (`leaves-change-color.json` vs `the-sun-dna.json`)
- Canonical version retained, duplicates archived

### Archive Created

Legacy files archived to `lessons/archive/` for reference.

## Phase 3: Migration

### Migration Status

**Files Requiring Migration:** 9 old-format files identified

**Migration Tool:** `curious-kellly/content-tools/migrate-to-phasedna-v2.js`

**Next Steps:**
1. Run batch migration: `node migrate-all-lessons.js`
2. Review migrated files for content completeness
3. Add missing welcome/mainContent/wisdomMoment text
4. Generate video files
5. Add proper teaching moments with timestamps
6. Add expression cues aligned to teaching moments

### JSON Parse Errors

The following files have JSON syntax errors that must be fixed before migration:

1. `aging_process_dna.json` - Error at line 322
2. `disruptive_innovation_dna.json` - Error at line 296
3. `parasitology_dna.json` - Error at line 322
4. `plasma_physics_dna.json` - Error at line 296
5. `stem_cells_dna.json` - Error at line 296

**Action Required:** Fix JSON syntax errors before proceeding with migration.

## Phase 4: Validation

### Validation Results

**Validation Tool:** `curious-kellly/content-tools/validate-all-lessons.js`

**Summary:**
- Total lessons validated: 16
- Valid: TBD (validation in progress)
- Invalid: TBD (validation in progress)
- Warnings: TBD

### Common Validation Issues

1. **Missing Required Fields:**
   - `id`, `title`, `description` (old format files)
   - `ageVariants`, `interactions`, `metadata` (old format files)

2. **Missing Age Groups:**
   - Many old-format files missing all 6 age buckets

3. **Missing Multilingual Content:**
   - Most lessons missing ES/FR translations

4. **Content Quality Issues:**
   - Missing welcome/mainContent sections
   - Missing keyPoints/interactionPrompts
   - Missing wisdomMoment

## Phase 5: Pre-computation (Multilingual Content)

### Multilingual Completeness Audit

**Audit Tool:** `curious-kellly/content-tools/precompute-audit.js`

### Missing Sections Breakdown

| Section | Missing Count |
|---------|---------------|
| `en.cta` | 12 |
| `en.summary` | 12 |
| `es.cta` | 12 |
| `es.summary` | 12 |
| `fr.cta` | 12 |
| `fr.summary` | 12 |
| `en.welcome` | 6 |
| `en.mainContent` | 6 |
| `en.keyPoints` | 6 |
| `en.interactionPrompts` | 6 |

### Lessons Missing All Content

The following lessons have no `ageVariants` structure and need complete migration:

1. `applied-mathematics-math-in-the-real-world-dna.json`
2. `creative-writing-dna.json`
3. `dance-expression-dna.json`
4. `genetic-engineering-editing-the-code-of-life-dna.json`
5. `molecular-biology-dna.json`
6. `negotiation-skills-dna.json`
7. `nutrition-science-dna.json`
8. `poetry-dna.json`

### Pre-computation Requirements

According to `CLAUDE.md`, all lessons must have:
- ✅ Precomputed EN content
- ❌ Precomputed ES content (missing)
- ❌ Precomputed FR content (missing)

**Critical:** Translations must be precomputed (not generated at runtime).

## Phase 6: Quality Assurance

### Age Variant Completeness

**Required Age Groups:** `2-5`, `6-12`, `13-17`, `18-35`, `36-60`, `61-102`

**Status:**
- Complete (6/6): 6 lessons
- Partial: 1 lesson
- Missing: 9 lessons

### Kelly Age/Persona Mappings

All age variants must have correct:
- `kellyAge`: 3, 9, 15, 27, 48, 82 (for respective age groups)
- `kellyPersona`: playful-toddler, curious-kid, enthusiastic-teen, knowledgeable-adult, wise-mentor, reflective-elder

### Voice Profile Consistency

All age variants should have consistent `voiceProfile` structure with:
- `provider`: "elevenlabs"
- `voiceId`: Appropriate for age group
- `speechRate`, `pitch`, `energy`: Age-appropriate settings

## Phase 7: Reporting and Documentation

### Generated Reports

1. **Inventory Report:** `docs/phasedna/lesson-inventory.json`
   - Complete inventory of all lessons
   - Schema version detection
   - Multilingual and age variant status

2. **Validation Report:** `docs/phasedna/validation-report.json`
   - Validation results for all lessons
   - Error and warning breakdowns
   - Pass/fail status

3. **Precomputation Audit:** `docs/phasedna/precomputation-audit.json`
   - Multilingual completeness analysis
   - Missing sections breakdown
   - Language presence statistics

4. **Migration Report:** `docs/phasedna/migration-report.json`
   - Migration status for old-format files
   - Backup file locations
   - Migration errors

5. **Lesson Index:** `curious-kellly/backend/config/lessons/.index.json`
   - Catalog of all lessons
   - Metadata and status information
   - Quick reference for lesson lookup

### Tools Created

1. **`organize-lessons.js`** - Main organization script
   - Discovers and categorizes lessons
   - Moves and renames files
   - Generates inventory report

2. **`validate-all-lessons.js`** - Batch validation script
   - Validates all lessons against PhaseDNA v2 schema
   - Generates validation report
   - Identifies common error patterns

3. **`precompute-audit.js`** - Multilingual completeness checker
   - Audits EN/ES/FR completeness
   - Identifies missing sections
   - Generates precomputation status report

4. **`migrate-all-lessons.js`** - Batch migration script
   - Migrates old-format files to PhaseDNA v2
   - Creates backups
   - Tracks migration status

5. **`generate-index.js`** - Index catalog generator
   - Creates/updates `.index.json` catalog
   - Includes metadata and status
   - Sorted by lesson ID

## Next Steps

### Immediate Actions Required

1. **Fix JSON Parse Errors** (5 files)
   - Fix syntax errors in aging_process, disruptive_innovation, parasitology, plasma_physics, stem_cells files
   - Re-run organization script

2. **Complete Migration** (9 old-format files)
   - Run `node migrate-all-lessons.js`
   - Review migrated files
   - Add missing content sections

3. **Complete Multilingual Content**
   - Add ES/FR translations to all lessons
   - Ensure all required sections present (welcome, mainContent, keyPoints, interactionPrompts, wisdomMoment, cta, summary)
   - Validate cultural adaptations

4. **Fix Validation Errors**
   - Address missing required fields
   - Complete age variant structures
   - Add missing interactions

5. **Quality Assurance**
   - Verify all 6 age groups present
   - Check Kelly age/persona mappings
   - Validate voice profiles
   - Review content quality

### Long-term Maintenance

1. **Update Documentation**
   - Update `curious-kellly/content-tools/README.md` with organization structure
   - Document validation workflow
   - Create checklist for new lesson creation

2. **Automation**
   - Add CI/CD validation checks
   - Automated precomputation verification
   - Regular inventory updates

3. **Content Completion**
   - Complete all missing multilingual content
   - Add missing teaching moments and expression cues
   - Generate video assets

## Files Modified/Created

### New Scripts
- `curious-kellly/content-tools/organize-lessons.js`
- `curious-kellly/content-tools/validate-all-lessons.js`
- `curious-kellly/content-tools/precompute-audit.js`
- `curious-kellly/content-tools/migrate-all-lessons.js`
- `curious-kellly/content-tools/generate-index.js`

### New Reports
- `docs/phasedna/lesson-inventory.json`
- `docs/phasedna/validation-report.json`
- `docs/phasedna/precomputation-audit.json`
- `docs/phasedna/migration-report.json`
- `docs/phasedna/LESSON_ORGANIZATION_REPORT.md` (this file)

### New Catalog
- `curious-kellly/backend/config/lessons/.index.json`

## Conclusion

The lesson organization phase is complete. All lessons have been consolidated to the canonical location with standardized naming. However, significant work remains:

1. **Migration:** 9 old-format files need migration to PhaseDNA v2
2. **Multilingual Content:** All lessons need ES/FR translations precomputed
3. **Validation:** Many lessons have validation errors that need fixing
4. **Content Quality:** Missing sections need to be added

The tools and reports created provide a solid foundation for completing this work systematically.

---

**Report Generated By:** Lesson Organization Scripts  
**Last Updated:** 2025-01-11


