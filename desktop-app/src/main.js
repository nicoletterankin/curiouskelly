const { app, BrowserWindow, Menu, shell, ipcMain } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');

// Keep a global reference of the window object
let mainWindow;

// Development mode flag
const isDev = process.argv.includes('--dev');

// App URL
const APP_URL = isDev 
  ? 'http://localhost:4321' 
  : 'https://curiouskelly.com';

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    backgroundColor: '#0a0a0b',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, '../build/icon.png'),
    show: false // Don't show until ready
  });

  // Load the app
  mainWindow.loadURL(APP_URL);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Check for updates (production only)
    if (!isDev) {
      autoUpdater.checkForUpdatesAndNotify();
    }
  });

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Create application menu
  createMenu();

  // Dev tools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Reload',
          accelerator: 'CmdOrCtrl+R',
          click: () => mainWindow.reload()
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => app.quit()
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Learn More',
          click: () => shell.openExternal('https://curiouskelly.com')
        },
        {
          label: 'Report Issue',
          click: () => shell.openExternal('https://curiouskelly.com/support')
        },
        { type: 'separator' },
        {
          label: 'About Curious Kelly',
          click: () => {
            const { dialog } = require('electron');
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Curious Kelly',
              message: 'Curious Kelly',
              detail: `Version: ${app.getVersion()}\n\nYour AI teacher for lifelong learning.\n\n© 2025 Lesson of the Day PBC`
            });
          }
        }
      ]
    }
  ];

  // macOS specific menu items
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// Auto-updater events
autoUpdater.on('update-available', () => {
  console.log('Update available');
});

autoUpdater.on('update-downloaded', () => {
  const { dialog } = require('electron');
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Update Ready',
    message: 'A new version of Curious Kelly is ready to install.',
    buttons: ['Restart Now', 'Later'],
    defaultId: 0
  }).then(result => {
    if (result.response === 0) {
      autoUpdater.quitAndInstall();
    }
  });
});

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Handle app-specific events
ipcMain.handle('app-version', () => {
  return app.getVersion();
});

ipcMain.handle('check-for-updates', () => {
  if (!isDev) {
    autoUpdater.checkForUpdates();
  }
});







