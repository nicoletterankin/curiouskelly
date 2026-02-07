require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  const r = await c.query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='lesson_atoms' ORDER BY ordinal_position");
  console.log('lesson_atoms columns:', JSON.stringify(r.rows, null, 2));
  
  // Also check unique constraints
  const constraints = await c.query("SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name='lesson_atoms'");
  console.log('constraints:', JSON.stringify(constraints.rows, null, 2));
  
  await c.end();
})();
