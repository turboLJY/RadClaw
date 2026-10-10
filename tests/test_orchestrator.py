"""Test orchestrator logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import AgentRole, PipelineState, ReviewIssue
from src.core.orchestrator import _determine_agents_to_run


def test_first_iteration_runs_all():
    state = PipelineState(session_id="test", iteration_count=0)
    state.iteration_count = 1
    agents = _determine_agents_to_run(state)
    assert len(agents) == 4
    assert agents[0] == AgentRole.VISUAL_ANALYSIS
    assert agents[-1] == AgentRole.QUALITY_REVIEW


def test_rerun_visual_issue():
    state = PipelineState(session_id="test", iteration_count=2)
    state.review_issues = [
        ReviewIssue(
            issue_id="RI001",
            issue_type="visual_inconsistency",
            description="test",
            attribution="visual",
            target_agent=AgentRole.VISUAL_ANALYSIS,
        ),
    ]
    agents = _determine_agents_to_run(state)
    assert AgentRole.VISUAL_ANALYSIS in agents
    assert AgentRole.QUALITY_REVIEW in agents
    assert AgentRole.EVIDENCE_RETRIEVAL not in agents


def test_rerun_generation_issue():
    state = PipelineState(session_id="test", iteration_count=2)
    state.review_issues = [
        ReviewIssue(
            issue_id="RI001",
            issue_type="overclaim",
            description="test",
            attribution="generation",
            target_agent=AgentRole.REPORT_WRITING,
        ),
    ]
    agents = _determine_agents_to_run(state)
    assert AgentRole.REPORT_WRITING in agents
    assert AgentRole.QUALITY_REVIEW in agents
    assert AgentRole.VISUAL_ANALYSIS not in agents


def test_no_issues_returns_empty():
    state = PipelineState(session_id="test", iteration_count=2)
    state.review_issues = []
    agents = _determine_agents_to_run(state)
    assert agents == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])