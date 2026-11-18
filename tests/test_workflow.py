from app.workflow.graph import (
    build_workflow,
    route_after_review,
)


def test_workflow_contains_all_expected_nodes():
    workflow = build_workflow()

    graph = workflow.get_graph()
    node_names = set(graph.nodes.keys())

    expected_nodes = {
        "smart_router",
        "planner",
        "researcher",
        "developer",
        "reviewer",
    }

    assert expected_nodes.issubset(node_names)


def test_approved_review_ends_workflow():
    state = {
        "review_status": "approved",
        "revision_count": 0,
        "max_revisions": 1,
    }

    assert route_after_review(state) == "end"


def test_changes_requested_routes_to_developer():
    state = {
        "review_status": "changes_requested",
        "revision_count": 0,
        "max_revisions": 1,
    }

    assert route_after_review(state) == "developer"


def test_revision_limit_ends_workflow():
    state = {
        "review_status": "changes_requested",
        "revision_count": 1,
        "max_revisions": 1,
    }

    assert route_after_review(state) == "end"


def test_missing_review_status_defaults_to_changes_requested():
    state = {
        "revision_count": 0,
        "max_revisions": 1,
    }

    assert route_after_review(state) == "developer"


def test_revision_limit_zero_prevents_retry():
    state = {
        "review_status": "changes_requested",
        "revision_count": 0,
        "max_revisions": 0,
    }

    assert route_after_review(state) == "end"