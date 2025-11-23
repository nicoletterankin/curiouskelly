# 🐠 Unity Export Instructions (Goldfish Edition)

## The Problem
Your files won't load on the website. They're stuck.

## The Fix
Turn off compression in Unity. Give us the raw files.

---

## 📋 Step-by-Step (Copy/Paste This)

**In Unity:**

1. Click **Edit** → **Project Settings**
2. Click **Player** (left sidebar)
3. Click the **WebGL** tab (looks like HTML5 icon)
4. Scroll down to **Publishing Settings**
5. Find **Compression Format**
6. Change it to **`Disabled`**
7. Click **File** → **Build Settings** → **Build**

---

## 📦 What to Send Us

After the build finishes, send us the **Build** folder.

We need these 3 files:
- `kelly-v1.data`
- `kelly-v1.wasm`
- `kelly-v1.framework.js`

**✅ Good:** Files end in `.data`, `.wasm`, `.js`  
**❌ Bad:** Files end in `.br` or `.gz`

---

## 🎯 That's It

If the files are **NOT** compressed, the website will work.

Current files = Broken because they're double-zipped.  
New files = Will work because they're raw.

