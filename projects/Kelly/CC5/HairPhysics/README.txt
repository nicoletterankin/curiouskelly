Kelly Hair Physics — Natural & Weighted
=====================================

Files included:
- Kelly_Hair_Physics.json
- Kelly_Hair_PhysicsMap.png   (grayscale weight map: black roots -> white tips)
- Fine_Strand_Noise.png       (fine noise map; use as secondary normal/bump)

Character Creator 5 — Import
----------------------------
1) Copy these files to: Documents\Reallusion\Custom\HairPhysics\
   (Or any folder you prefer. The JSON references the PNGs by filename.)
2) In CC5, select your Hair mesh in the Scene Manager.
3) Open Modify Panel ▸ Physics ▸ Load Preset.
4) Choose 'Kelly_Hair_Physics.json'.
5) If prompted for images, point to Kelly_Hair_PhysicsMap.png and Fine_Strand_Noise.png.
6) Press Alt + Space to preview real-time simulation.
7) Tweak 'Elasticity' (+/- 0.05) and 'Damping' (+/- 0.05) to taste.

iClone 8 — Usage
----------------
1) Send the character to iClone (Shift + F12).
2) Enable Soft Cloth for Hair and load the same preset if needed.
3) Create a Wind node (Direction ~ -10°, Strength 1.5) for natural daily motion.
4) Cache the simulation before final rendering for stability.

Notes
-----
- Weight map assumes UVs oriented with roots at the top of the texture. If your UVs differ,
  rotate or flip the PNG inside your image editor.
- For stronger tips movement, lower 'airResistance' slightly (e.g., 0.05).
- For stiffer hair (less sway), increase 'damping' to ~0.55.
