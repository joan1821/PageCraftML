# Deploy PageCraftML to Railway

## Backend Server

This NN server connects to:
`https://pagecraftserver-production-d2dd.up.railway.app`

## Quick Deploy

### Option 1: GitHub

1. Push to GitHub
   ```bash
   git add .
   git commit -m "Ready for Railway"
   git push origin main
   ```

2. Deploy on Railway
   - Visit [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `PageCraftML`

### Option 2: Railway CLI

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

## API Endpoints

- `GET /` - Status check
- `GET /health` - Health check
- `POST /process` - Main API endpoint

## Test Deployment

```bash
curl https://YOUR_RAILWAY_URL/health

curl -X POST https://YOUR_RAILWAY_URL/process \
  -H "Content-Type: application/json" \
  -d '{
    "itemsByResolution": {
      "Desktop (1920x1080)": [
        {"id": "1", "width": 1920, "height": 1080}
      ],
      "Mobile (375x667)": []
    }
  }'
```

## Monitor

- Logs: Railway Dashboard → Your Service → Logs
- Metrics: Railway Dashboard → Metrics
- Health: Visit your-url/health

## Troubleshooting

- Build fails: Check Railway logs
- App not responding: Check deployment status and /health endpoint
- Port issues: Railway handles PORT automatically
