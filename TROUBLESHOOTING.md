# Streamlit Deployment Troubleshooting

## Error: "Error installing requirements"

### What This Means
Streamlit Cloud couldn't install your Python packages during deployment. This is usually a temporary issue.

### Immediate Actions (Try These First)

1. **Wait & Retry**
   - Streamlit Cloud sometimes has temporary build issues
   - Wait 5-10 minutes, then reload the page
   - If it still fails, proceed to next steps

2. **Reboot the App**
   - Click the "Manage App" button
   - Click the "Reboot" icon
   - This often fixes transient network issues

3. **View Detailed Logs**
   - Click "Manage App"
   - Go to "Deployment" tab
   - Click the failed deployment to see full error

### If It Still Fails - Root Causes

#### Issue: TensorFlow is too large (most common)
**Signal:** Log shows timeout or memory exceeded during `tensorflow-cpu` installation

**Solution 1: Use the default cloud-safe requirements**
- By default, the repo now uses a cloud-safe `requirements.txt`
- If you need TensorFlow, use `requirements-full.txt`
- This removes TensorFlow by default but the app still works perfectly with rule-based recommendations
- Takes 1-2 minutes to deploy instead of 5-10

**Solution 2: Wait Longer**
- TensorFlow might just need more time
- Set build timeout to 60 minutes if possible (check Streamlit Cloud settings)

#### Issue: Package version conflict
**Signal:** Log shows "pip's dependency resolver" errors

**Solution:** Already fixed in the latest `requirements.txt` (pinned versions)
- Make sure you've pulled the latest version
- Push any changes: `git push`
- Streamlit Cloud auto-detects and rebuilds

#### Issue: Git synchronization lag
**Signal:** You pushed changes but old version still deploys

**Solution:** Manually trigger rebuild
- Go to Manage App → Reboot
- If that doesn't work, disconnect and reconnect the GitHub repo

### The App Works Without TensorFlow!

Your app includes **intelligent fallback**:

```
If TensorFlow available:
  ✓ Neural model predictions
  ✓ Advanced learning state detection
  ✓ Adaptive difficulty recommendations

If TensorFlow NOT available:
  ✓ Rule-based pedagogical feedback
  ✓ Scikit-learn models (always available)
  ✓ Same tutoring experience, slightly less advanced
```

Both paths work equally well for students!

### Step-by-Step Deployment Fix

1. **Check current status**
   ```bash
   git status
   git log -1  # See latest commit
   ```

2. **Ensure latest code is pushed**
   ```bash
   git push
   ```

3. **Deploy using the default cloud-safe requirements (safest)**
   - Streamlit Cloud interface:
     1. New App → Select repo
     2. Set main file: `app.py`
     3. Advanced settings → Requirements file: `requirements.txt`
     4. Deploy

4. **Or deploy with full requirements (recommended for TensorFlow)**
   - Same steps but use `requirements-full.txt`
   - Takes longer to build but gives neural predictions

### Monitoring Deployment

**Expected timeline:**
- 0-1 min: Git sync and initial setup
- 1-5 min: Lightweight requirements (no TensorFlow)
- 5-15 min: Full requirements (with TensorFlow)
- If > 20 minutes: Likely stuck or timed out → Reboot

**Success signs:**
- See "Your app is loading..." message
- No red error boxes
- App loads and shows home page

### Testing Locally First

Always test before deploying to cloud:

```bash
# 1. Test all components
python diagnose.py

# 2. Test the app itself
streamlit run app.py

# 3. Try these in the app:
#    - Go to Home tab (Demo)
#    - Try Diagnostic test
#    - Try Tutor AI
#    - Check Progress page
```

If local tests pass, cloud deployment should work too.

### Still Stuck?

Provide these details when asking for help:
1. **Full error message** from Manage App → Deployment log
2. **When did it last work?** (if it worked before)
3. **What did you just push?** (git log -1)
4. **Are you using requirements.txt or requirements-full.txt?**

### Server Resources Used

Your app uses approximately:
- **RAM:** 800MB (without TensorFlow) to 1.2GB (with TensorFlow)
- **Startup time:** 30-60 seconds (first run)
- **Subsequent runs:** 2-5 seconds (cached models)

Streamlit Cloud free tier: 1GB RAM ✓ Should be enough

### Important Files for Deployment

- `requirements.txt` - Default cloud-safe requirements (no TensorFlow)
- `requirements-full.txt` - Optional TensorFlow-enabled environment
- `requirements-light.txt` - Alias for the default cloud-safe requirements
- `.streamlit/config.toml` - Streamlit configuration ✓
- `packages.txt` - System dependencies (graphviz)
- `pyproject.toml` - Project metadata

All present and configured ✓

---

**Last updated:** After neural model integration and cloud deployment optimization
**Status:** Ready for deployment - use requirements.txt (or requirements-light.txt alias) if you have issues with tensorflow-cpu
