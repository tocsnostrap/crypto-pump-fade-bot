import { getAuthenticatedUser, createRepository } from './server/github';
import { execSync } from 'child_process';

async function exportToGitHub() {
  const repoName = 'crypto-pump-fade-bot';
  const description = 'Automated cryptocurrency trading bot for Gate.io and Bitget futures - detects pumps and enters short positions on reversal signals';
  
  console.log('Getting GitHub user...');
  const user = await getAuthenticatedUser();
  console.log(`Authenticated as: ${user.login}`);
  
  console.log('Creating/getting repository...');
  const repo = await createRepository(repoName, description, false);
  console.log(`Repository URL: ${repo.html_url}`);
  
  console.log('Setting up git remote...');
  try {
    execSync('git remote remove origin 2>/dev/null || true', { stdio: 'inherit' });
  } catch {}
  
  const remoteUrl = `https://github.com/${user.login}/${repoName}.git`;
  execSync(`git remote add origin ${remoteUrl}`, { stdio: 'inherit' });
  
  console.log(`\n✅ Repository ready: ${repo.html_url}`);
  console.log('\nTo push your code, run:');
  console.log('  git push -u origin main');
  console.log('\nOr if your branch is named differently:');
  console.log('  git branch -M main && git push -u origin main');
}

exportToGitHub().catch(console.error);
