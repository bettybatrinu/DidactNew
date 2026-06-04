# Streamlit Cloud Deployment Guide

## Quick Deploy

1. **Push to GitHub** (already done ✓)
   ```bash
   git push
   ```

2. **Go to Streamlit Cloud** 
   - Visit: https://streamlit.io/cloud
   - Click "New app"
   - Select this repository
   - Set main file path to: `app.py`
   - Click "Deploy"

## If You See "Error installing requirements"

### Solution 1: Check Status (First Try)
- Click **"Manage App"** button on the error page
- Look at the **Deployment** tab for details
- Common reasons:
  - Network timeout during TensorFlow download (too large)
  - Memory constraints during build
  - Package version conflicts

### Solution 2: Restart Deployment
- Click **"Reboot app"** in the Manage App menu
- This sometimes fixes transient network issues

### Solution 3: View Build Logs
In **Manage App → Deployment** tab:
- Click on the failed deployment
- Look for the exact error message
- Common issues:
  ```
  ERROR: pip's dependency resolver does not currently take into account
  ```
  This means a package version conflict. We're handling this now.

### Solution 4: Alternative - Lighter Requirements
If tensorflow-cpu still fails, use this lighter version:

Edit `requirements.txt` to:
```
streamlit==1.35.0
pandas==2.1.4
numpy==1.24.3
openpyxl==3.1.1
scikit-learn==1.3.2
joblib==1.3.2
# Optional: remove tensorflow-cpu for faster cloud deployment
# The app will still work with fallback to rule-based recommendations
```

The app is designed to **gracefully degrade** - it works without TensorFlow!

### Solution 5: Clear Streamlit Cache (User Side)
Users can clear their browser cache:
1. Streamlit → Settings (top right)
2. "Clear cache"
3. Refresh the page

## File Structure for Cloud

Your repository needs these files for Streamlit Cloud:
```
repo/
├── app.py                    # ✓ Main app file
├── requirements.txt          # ✓ Python dependencies (fixed)
├── .streamlit/config.toml    # ✓ Streamlit configuration
├── .streamlit/secrets.toml   # ✓ Secrets (currently empty)
├── packages.txt              # ✓ System dependencies (graphviz)
├── data/processed/           # ✓ Datasets
├── models/                   # ✓ Pre-trained models
└── src/                      # ✓ Python modules
```

All these are present ✓

## Performance Optimization

The current setup uses:
- `tensorflow-cpu` (smaller than GPU version)
- Pre-trained models (no training on deployment)
- Cached asset loading
- Pandas operations (efficient)

This should work fine on Streamlit Cloud's free tier (1GB RAM).

## Next Steps

1. **Wait 5-10 minutes** after pushing - Streamlit Cloud takes time to detect changes
2. **Try the deployment link** again
3. If it still fails → Copy the **exact error message** and share it
4. The app will show helpful error messages if something fails

## Local Testing (Confirm Before Deploy)

```bash
cd /workspaces/didactai
python diagnose.py          # Check all components
streamlit run app.py        # Test locally
```

If both pass, the cloud deployment should work.

## Architecture

The app includes graceful fallback:
- ✓ Neural model available → use neural predictions
- ✓ Neural model unavailable → use rule-based feedback
- Either way, the tutor experience works!

---

**Last updated:** After neural model integration
**Status:** Ready for Streamlit Cloud deployment
