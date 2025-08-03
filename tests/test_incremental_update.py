import os
from types import SimpleNamespace

from processing.knowledge_graph import DocumentTracker


def make_doc(name: str, content: str, mtime: float = 0.0):
    return SimpleNamespace(content=content, meta={"filename": name, "modified_date": mtime})


def test_document_tracker_detects_changes(tmp_path):
    state_file = tmp_path / "state.json"

    tracker = DocumentTracker(str(state_file))
    docs = [make_doc("a.txt", "hello", 1), make_doc("b.txt", "world", 2)]
    to_process, deleted = tracker.filter_documents(docs)
    assert len(to_process) == 2
    assert deleted == []
    tracker.save()

    tracker = DocumentTracker(str(state_file))
    docs2 = [make_doc("a.txt", "hello", 1), make_doc("b.txt", "world!", 2)]
    to_process, deleted = tracker.filter_documents(docs2)
    assert len(to_process) == 1
    assert to_process[0].meta["filename"] == "b.txt"
    assert deleted == []
    tracker.save()

    tracker = DocumentTracker(str(state_file))
    docs3 = [make_doc("a.txt", "hello", 1)]
    to_process, deleted = tracker.filter_documents(docs3)
    assert to_process == []
    assert deleted == ["b.txt"]
