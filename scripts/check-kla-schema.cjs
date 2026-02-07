require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  const r = await c.query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='kelly_lesson_assets' ORDER BY ordinal_position");
  r.rows.forEach(x => console.log(x.column_name, '-', x.data_type));
  console.log('\nTotal columns:', r.rows.length);
  
  // Check day_number type
  const dt = r.rows.find(x => x.column_name === 'day_number');
  if (dt) console.log('day_number type:', dt.data_type);
  
  // Check phase type
  const pt = r.rows.find(x => x.column_name === 'phase');
  if (pt) console.log('phase type:', pt.data_type);
  
  await c.end();
})();
