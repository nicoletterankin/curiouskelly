# Unity Demo Environment Notes

Add the following keys to `.env` (or to the hosting provider’s secret manager) so the `/demo/avatar/` page can stream real lesson assets:

```
PUBLIC_UNITY_IFRAME_SRC=https://assets.curiouskelly.com/unity/kelly-v1/index.html
PUBLIC_UNITY_SAMPLE_JSON=https://assets.curiouskelly.com/unity/kelly-v1/content/water-cycle.a2f.json
PUBLIC_UNITY_SAMPLE_AUDIO=https://assets.curiouskelly.com/unity/kelly-v1/audio/water-cycle-18-35.mp3
PUBLIC_UNITY_SAMPLE_EXPRESSIONS=https://assets.curiouskelly.com/unity/kelly-v1/content/water-cycle.expressions.json
```

During local development you can leave the values blank to disable the “Play sample lesson” button or point them at files served from `public/unity/kelly-v1/`.

All URLs must be publicly readable (no signed query strings) and return with `Access-Control-Allow-Origin: *` or the iframe will fail to download the assets.


