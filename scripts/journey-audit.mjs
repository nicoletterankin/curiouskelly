import puppeteer from 'puppeteer';

async function auditJourney() {
  console.log('🗺️ Starting Journey Navigation Audit...\n');
  
  const browser = await puppeteer.launch({ 
    headless: true, 
    args: ['--no-sandbox', '--disable-setuid-sandbox'] 
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  
  try {
    // Go to app
    await page.goto('http://localhost:3000/learn.html?testing=true&bypass=testing', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    await new Promise(r => setTimeout(r, 1500));
    
    // Click Journey in nav
    await page.click('.nav-item[data-scene="journey"]');
    await new Promise(r => setTimeout(r, 800));
    
    // Screenshot Calendar tab
    await page.screenshot({ path: 'test-screenshots/journey-1-calendar.png', fullPage: false });
    console.log('📅 1. Calendar tab captured');
    
    // Analyze calendar grid
    const calendarInfo = await page.evaluate(() => {
      const cells = document.querySelectorAll('.day-cell');
      const gridView = document.getElementById('grid-view');
      const months = document.querySelectorAll('.month-section');
      return {
        totalCells: cells.length,
        monthsVisible: months.length,
        gridViewHtml: gridView?.innerHTML?.substring(0, 500) || 'NOT FOUND',
        hasImages: Array.from(cells).some(c => c.querySelector('img')),
        hasTitles: Array.from(cells).some(c => c.textContent?.length > 3)
      };
    });
    console.log('   Calendar grid:', calendarInfo);
    
    // Click Week tab
    await page.click('.journey-tab[data-tab="week"]');
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: 'test-screenshots/journey-2-week.png', fullPage: false });
    console.log('📆 2. Week tab captured');
    
    // Analyze week view
    const weekInfo = await page.evaluate(() => {
      const weekDays = document.getElementById('week-days');
      const dayCards = weekDays?.querySelectorAll('.week-day') || [];
      return {
        totalDays: dayCards.length,
        dayContent: Array.from(dayCards).slice(0, 3).map(d => ({
          hasTitle: !!d.querySelector('.week-day-topic, .week-day-title'),
          text: d.textContent?.substring(0, 100).trim()
        }))
      };
    });
    console.log('   Week view:', weekInfo);
    
    // Click Curriculum tab
    await page.click('.journey-tab[data-tab="curriculum"]');
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ path: 'test-screenshots/journey-3-curriculum.png', fullPage: false });
    console.log('📚 3. Curriculum tab captured');
    
    // Test search functionality
    const searchInput = await page.$('#curriculum-search-input');
    if (searchInput) {
      await searchInput.type('dream');
      await new Promise(r => setTimeout(r, 800));
      await page.screenshot({ path: 'test-screenshots/journey-4-search.png', fullPage: false });
      console.log('🔍 4. Search results captured');
      
      // Analyze search results
      const searchResults = await page.evaluate(() => {
        const results = document.querySelectorAll('.curriculum-lesson, .search-result');
        return {
          resultCount: results.length,
          content: Array.from(results).slice(0, 3).map(r => r.textContent?.substring(0, 80))
        };
      });
      console.log('   Search results:', searchResults);
    }
    
    // Click Saved/Bookmarks tab
    await page.click('.journey-tab[data-tab="bookmarks"]');
    await new Promise(r => setTimeout(r, 500));
    await page.screenshot({ path: 'test-screenshots/journey-5-bookmarks.png', fullPage: false });
    console.log('🔖 5. Bookmarks tab captured');
    
    // Go back to calendar and click a day cell
    await page.click('.journey-tab[data-tab="calendar"]');
    await new Promise(r => setTimeout(r, 600));
    
    // Click a day cell to see phase selector
    const dayCells = await page.$$('.day-cell:not(.locked)');
    if (dayCells.length > 5) {
      await dayCells[5].click();
      await new Promise(r => setTimeout(r, 600));
      await page.screenshot({ path: 'test-screenshots/journey-6-phase-selector.png', fullPage: false });
      console.log('🎯 6. Phase selector modal captured');
      
      // Analyze phase selector
      const phaseSelectorInfo = await page.evaluate(() => {
        const modal = document.getElementById('phase-selector-modal');
        const topic = document.getElementById('phase-selector-topic');
        const phases = document.querySelectorAll('.phase-option');
        return {
          isVisible: modal?.classList.contains('active'),
          topic: topic?.textContent,
          phaseCount: phases.length,
          phaseNames: Array.from(phases).map(p => p.querySelector('.phase-option-name')?.textContent)
        };
      });
      console.log('   Phase selector:', phaseSelectorInfo);
    }
    
    console.log('\n✅ Journey audit complete!');
    console.log('Screenshots saved to test-screenshots/journey-*.png');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    await page.screenshot({ path: 'test-screenshots/journey-error.png', fullPage: false });
  } finally {
    await browser.close();
  }
}

auditJourney();

