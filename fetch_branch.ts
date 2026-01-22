import { getUncachableGitHubClient, getAuthenticatedUser } from './server/github';
import * as fs from 'fs';
import * as path from 'path';

const REPO_NAME = 'crypto-pump-fade-bot';
const BRANCH_NAME = 'cursor/live-trading-strategy-improvement-b2a2';

async function fetchBranch() {
  const octokit = await getUncachableGitHubClient();
  const user = await getAuthenticatedUser();
  const owner = user.login;
  
  console.log(`Fetching branch: ${BRANCH_NAME} from ${owner}/${REPO_NAME}...`);
  
  try {
    // Get the branch
    const { data: branch } = await octokit.repos.getBranch({
      owner,
      repo: REPO_NAME,
      branch: BRANCH_NAME,
    });
    console.log(`Branch found! Commit: ${branch.commit.sha}`);
    
    // Get the tree
    const { data: tree } = await octokit.git.getTree({
      owner,
      repo: REPO_NAME,
      tree_sha: branch.commit.sha,
      recursive: 'true',
    });
    
    console.log(`\nFiles in branch (${tree.tree.length} items):`);
    for (const item of tree.tree) {
      if (item.type === 'blob') {
        console.log(`  ${item.path}`);
      }
    }
    
    return tree.tree.filter(item => item.type === 'blob');
  } catch (error: any) {
    console.error(`Error: ${error.message}`);
    if (error.status === 404) {
      console.log('\nBranch not found. Available branches:');
      const { data: branches } = await octokit.repos.listBranches({
        owner,
        repo: REPO_NAME,
      });
      branches.forEach(b => console.log(`  - ${b.name}`));
    }
    return null;
  }
}

fetchBranch().catch(console.error);
