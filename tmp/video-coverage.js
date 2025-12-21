const { createClient } = require('@supabase/supabase-js');

const url = 'https://tvjalxxsyryjphkforjv.supabase.co';
const key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(url, key, { auth: { persistSession: false } });

const PAGE_SIZE = 1000;

function normalizeDay(row) {
  return row.day_number ?? row.lesson_day ?? null;
}

async function fetchAllVideoRows() {
  const rows = [];
  let from = 0;
  while (true) {
    const { data, error } = await supabase
      .from('kelly_video_assets')
      .select('day_number, lesson_day, phase, template, language')
      .eq('asset_type', 'video')
      .range(from, from + PAGE_SIZE - 1);
    if (error) {
      throw error;
    }
    if (!data || data.length === 0) {
      break;
    }
    rows.push(...data);
    if (data.length < PAGE_SIZE) {
      break;
    }
    from += PAGE_SIZE;
  }
  return rows;
}

function summarize(rows) {
  const dayMap = new Map();
  for (const row of rows) {
    const day = normalizeDay(row);
    if (!day) continue;
    if (!dayMap.has(day)) {
      dayMap.set(day, { count: 0, phases: new Set() });
    }
    const entry = dayMap.get(day);
    entry.count += 1;
    if (row.phase) entry.phases.add(row.phase);
  }
  return dayMap;
}

async function main() {
  const rows = await fetchAllVideoRows();
  const dayMap = summarize(rows);

  const targetDays = [1, 50, 100, 200, 300, 365];
  const targetSummary = targetDays.map((day) => {
    const entry = dayMap.get(day);
    return {
      day,
      clips: entry ? entry.count : 0,
      uniquePhases: entry ? entry.phases.size : 0,
    };
  });

  const allDays = Array.from(dayMap.entries()).map(([day, entry]) => ({
    day: Number(day),
    clips: entry.count,
    uniquePhases: entry.phases.size,
  })).sort((a, b) => a.day - b.day);

  const daysWithAny = allDays.length;
  const daysWithSeven = allDays.filter((d) => d.uniquePhases >= 7).length;

  console.log(JSON.stringify({
    totalRows: rows.length,
    targetSummary,
    daysWithAny,
    daysWithSeven,
    allDays,
  }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});



