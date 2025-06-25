import logging
from typing import Any, Dict, List, Optional
from github import Github, GithubException
from mcp import Tool
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class GitHubToolBase(BaseModel):
    """Base class for GitHub tool parameters."""
    pass


class CreateIssueParams(GitHubToolBase):
    repo: str = Field(..., description="Repository in format 'owner/repo'")
    title: str = Field(..., description="Issue title")
    body: Optional[str] = Field(None, description="Issue description")
    assignees: Optional[List[str]] = Field(None, description="List of usernames to assign")
    labels: Optional[List[str]] = Field(None, description="List of labels to apply")


class CreatePRParams(GitHubToolBase):
    repo: str = Field(..., description="Repository in format 'owner/repo'")
    title: str = Field(..., description="PR title")
    body: Optional[str] = Field(None, description="PR description")
    head: str = Field(..., description="Branch containing changes")
    base: str = Field("main", description="Base branch")
    draft: bool = Field(False, description="Create as draft PR")


class ListIssuesParams(GitHubToolBase):
    repo: str = Field(..., description="Repository in format 'owner/repo'")
    state: str = Field("open", description="Issue state: open, closed, or all")
    assignee: Optional[str] = Field(None, description="Filter by assignee username")
    labels: Optional[List[str]] = Field(None, description="Filter by labels")
    limit: int = Field(30, description="Maximum number of issues to return")


class GitHubTools:
    """Collection of GitHub integration tools."""
    
    def __init__(self):
        if not settings.validate_github():
            raise ValueError("GitHub token not configured")
        self.github = Github(settings.github_token)
    
    def create_issue_tool(self) -> Tool:
        """Create a tool for creating GitHub issues."""
        async def create_issue(params: Dict[str, Any]) -> str:
            args = CreateIssueParams(**params)
            try:
                repo = self.github.get_repo(args.repo)
                issue = repo.create_issue(
                    title=args.title,
                    body=args.body or "",
                    assignees=args.assignees or [],
                    labels=args.labels or []
                )
                return f"Created issue #{issue.number}: {issue.html_url}"
            except GithubException as e:
                logger.error(f"Failed to create issue: {e}")
                return f"Error creating issue: {str(e)}"
        
        return Tool(
            name="github_create_issue",
            description="Create a new GitHub issue",
            inputSchema=CreateIssueParams.model_json_schema(),
            handler=create_issue
        )
    
    def create_pr_tool(self) -> Tool:
        """Create a tool for creating pull requests."""
        async def create_pr(params: Dict[str, Any]) -> str:
            args = CreatePRParams(**params)
            try:
                repo = self.github.get_repo(args.repo)
                pr = repo.create_pull(
                    title=args.title,
                    body=args.body or "",
                    head=args.head,
                    base=args.base,
                    draft=args.draft
                )
                return f"Created PR #{pr.number}: {pr.html_url}"
            except GithubException as e:
                logger.error(f"Failed to create PR: {e}")
                return f"Error creating PR: {str(e)}"
        
        return Tool(
            name="github_create_pr",
            description="Create a new GitHub pull request",
            inputSchema=CreatePRParams.model_json_schema(),
            handler=create_pr
        )
    
    def list_issues_tool(self) -> Tool:
        """Create a tool for listing GitHub issues."""
        async def list_issues(params: Dict[str, Any]) -> str:
            args = ListIssuesParams(**params)
            try:
                repo = self.github.get_repo(args.repo)
                issues = repo.get_issues(
                    state=args.state,
                    assignee=args.assignee,
                    labels=args.labels or []
                )
                
                issue_list = []
                for i, issue in enumerate(issues[:args.limit]):
                    issue_list.append(
                        f"#{issue.number}: {issue.title} ({issue.state})"
                    )
                
                return "\n".join(issue_list) if issue_list else "No issues found"
            except GithubException as e:
                logger.error(f"Failed to list issues: {e}")
                return f"Error listing issues: {str(e)}"
        
        return Tool(
            name="github_list_issues",
            description="List GitHub issues with filters",
            inputSchema=ListIssuesParams.model_json_schema(),
            handler=list_issues
        )
    
    def get_all_tools(self) -> List[Tool]:
        """Get all GitHub tools."""
        return [
            self.create_issue_tool(),
            self.create_pr_tool(),
            self.list_issues_tool()
        ]