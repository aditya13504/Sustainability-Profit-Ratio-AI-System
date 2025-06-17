# Vercel Deployment Guide for Enhanced SPR Analyzer

## Overview
This guide provides step-by-step instructions to deploy the Enhanced SPR Analyzer dashboard on Vercel, a serverless platform that's perfect for Python web applications.

## Prerequisites
- GitHub account
- Vercel account (free tier available)
- API keys for the services you want to use

## Step 1: Prepare Your Repository

### 1.1 Fork or Upload to GitHub
1. **Option A - Fork**: If this is a public repository, fork it to your GitHub account
2. **Option B - Upload**: 
   - Create a new repository on GitHub
   - Upload your project files (excluding `.env` file)

### 1.2 Verify Required Files
Ensure these files are in your repository root:
- ✅ `app.py` (Vercel entry point)
- ✅ `vercel.json` (Vercel configuration)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.env.production` (Environment template)
- ✅ `src/` directory with dashboard code

## Step 2: Deploy on Vercel

### 2.1 Connect Repository
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"New Project"**
3. Import your GitHub repository
4. Select the repository containing your SPR Analyzer

### 2.2 Configure Build Settings
Vercel should automatically detect this as a Python project:
- **Framework Preset**: Other
- **Build Command**: (leave empty - automatic)
- **Output Directory**: (leave empty - automatic)
- **Install Command**: `pip install -r requirements.txt`

### 2.3 Set Environment Variables
In the Vercel dashboard, add these environment variables:

#### Required for Basic Functionality:
```
GEMINI_API_KEY=your_gemini_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

#### Optional (Enhanced Features):
```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key
NVIDIA_CUOPT_API_KEY=your_nvidia_cuopt_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
```

#### System Variables:
```
PYTHONPATH=.
FLASK_ENV=production
DEBUG=false
```

### 2.4 Deploy
1. Click **"Deploy"**
2. Wait for the build to complete (2-5 minutes)
3. Your dashboard will be available at `https://your-project-name.vercel.app`

## Step 3: Get Required API Keys

### 3.1 Google Gemini API (Required)
1. Go to [Google AI Studio](https://makersuite.google.com)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add to Vercel environment variables

### 3.2 Mistral AI API (Required for Chat)
1. Go to [Mistral AI Console](https://console.mistral.ai)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add to Vercel environment variables

### 3.3 Financial Data APIs (Optional)
- **Alpha Vantage**: [Get free key](https://www.alphavantage.co/support/#api-key) (5 requests/min)
- **Finnhub**: [Get free key](https://finnhub.io/register) (60 requests/min)
- **NewsAPI**: [Get free key](https://newsapi.org/) (1,000 requests/day)

### 3.4 NVIDIA cuOpt (Optional)
1. Go to [NVIDIA Developer Portal](https://developer.nvidia.com/cuopt)
2. Apply for API access
3. Once approved, get your API key

## Step 4: Verify Deployment

### 4.1 Check Application Status
1. Open your Vercel dashboard
2. Check deployment logs for any errors
3. Visit your live URL

### 4.2 Test Core Features
1. **Homepage**: Should load without errors
2. **Company Search**: Try searching for a company
3. **Analysis**: Run a basic analysis
4. **AI Chat**: Test the Mistral AI assistant

### 4.3 Monitor Performance
- Check Vercel function logs for any timeout issues
- Monitor API usage in respective provider dashboards

## Step 5: Troubleshooting

### Common Issues

#### Build Failures
- **Issue**: Module not found errors
- **Solution**: Check `requirements.txt` includes all dependencies

#### Timeout Errors
- **Issue**: Function timeout (60 seconds max on Vercel)
- **Solution**: Reduce analysis scope or optimize code

#### API Rate Limits
- **Issue**: Too many API requests
- **Solution**: Implement caching or reduce request frequency

#### Environment Variables Not Working
- **Issue**: API keys not recognized
- **Solution**: Verify environment variable names match exactly

### Debug Steps
1. Check Vercel function logs in dashboard
2. Verify all environment variables are set
3. Test API keys individually
4. Check file paths are correct

## Step 6: Custom Domain (Optional)

### 6.1 Add Custom Domain
1. In Vercel dashboard, go to project settings
2. Click **"Domains"**
3. Add your custom domain
4. Follow DNS configuration instructions

### 6.2 SSL Certificate
Vercel automatically provides SSL certificates for all domains.

## Step 7: Updates and Maintenance

### 7.1 Automatic Deployments
- Push changes to your GitHub repository
- Vercel automatically rebuilds and deploys
- No manual intervention required

### 7.2 Environment Updates
- Update API keys in Vercel dashboard as needed
- Changes take effect on next deployment

### 7.3 Monitoring
- Check Vercel analytics for usage patterns
- Monitor API usage to avoid rate limits
- Review function logs periodically

## Cost Considerations

### Vercel Free Tier Includes:
- 100GB bandwidth per month
- 100GB hours of function execution
- Unlimited static file serving
- Custom domains with SSL

### API Costs:
- **Gemini**: Free tier with generous limits
- **Mistral**: Pay-per-use pricing
- **Financial APIs**: Most have free tiers

## Support Resources

- **Vercel Documentation**: [docs.vercel.com](https://docs.vercel.com)
- **Python on Vercel**: [vercel.com/docs/functions/serverless-functions/runtimes/python](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- **Dashboard Issues**: Check GitHub repository issues

## Security Best Practices

1. **Never commit API keys** to your repository
2. **Use environment variables** for all sensitive data
3. **Regularly rotate API keys**
4. **Monitor API usage** for unusual activity
5. **Keep dependencies updated**

---

Your Enhanced SPR Analyzer should now be successfully deployed on Vercel! 🚀

The dashboard provides a comprehensive sustainability-profit-research analysis platform with AI-powered insights, accessible from anywhere with a modern web browser.
