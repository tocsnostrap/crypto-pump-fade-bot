import { getUncachableGitHubClient, getAuthenticatedUser } from './server/github';
import * as fs from 'fs';
import * as path from 'path';

const REPO_NAME = 'crypto-pump-fade-bot';

const FILES_TO_PUSH = [
  'main.py',
  'bot_config.json',
  'start.sh',
  'start_all.sh',
  'backtest.py',
  'backtest_compare.py',
  'analyze_winners.py',
  'pattern_analysis.py',
  'package.json',
  'tsconfig.json',
  'pyproject.toml',
  'replit.md',
  '.replit',
  'server/index.ts',
  'server/routes.ts',
  'server/storage.ts',
  'server/static.ts',
  'server/vite.ts',
  'server/github.ts',
  'client/src/App.tsx',
  'client/src/main.tsx',
  'client/src/index.css',
  'client/src/pages/dashboard.tsx',
  'client/src/pages/not-found.tsx',
  'client/src/lib/queryClient.ts',
  'client/src/lib/utils.ts',
  'client/index.html',
  'shared/schema.ts',
  'script/build.ts',
  'vite.config.ts',
  'tailwind.config.ts',
  'drizzle.config.ts',
  'postcss.config.js',
];

async function pushToGitHub() {
  const octokit = await getUncachableGitHubClient();
  const user = await getAuthenticatedUser();
  const owner = user.login;
  
  console.log(`Pushing files to ${owner}/${REPO_NAME}...`);
  
  let successCount = 0;
  let errorCount = 0;
  
  for (const filePath of FILES_TO_PUSH) {
    try {
      const fullPath = path.join(process.cwd(), filePath);
      
      if (!fs.existsSync(fullPath)) {
        console.log(`⚠️  Skipping ${filePath} (not found)`);
        continue;
      }
      
      const content = fs.readFileSync(fullPath);
      const base64Content = content.toString('base64');
      
      let sha: string | undefined;
      try {
        const { data } = await octokit.repos.getContent({
          owner,
          repo: REPO_NAME,
          path: filePath,
        });
        if (!Array.isArray(data) && 'sha' in data) {
          sha = data.sha;
        }
      } catch {
        // File doesn't exist yet
      }
      
      await octokit.repos.createOrUpdateFileContents({
        owner,
        repo: REPO_NAME,
        path: filePath,
        message: `Add ${filePath}`,
        content: base64Content,
        sha,
      });
      
      console.log(`✅ Pushed: ${filePath}`);
      successCount++;
    } catch (error: any) {
      console.error(`❌ Error pushing ${filePath}: ${error.message}`);
      errorCount++;
    }
  }
  
  console.log(`\n========================================`);
  console.log(`✅ Successfully pushed: ${successCount} files`);
  console.log(`❌ Errors: ${errorCount} files`);
  console.log(`\n🔗 Repository: https://github.com/${owner}/${REPO_NAME}`);
}

pushToGitHub().catch(console.error);
