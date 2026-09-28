CREATE TABLE IF NOT EXISTS study_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS study_items (
    item_index INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL CHECK(split IN ('train', 'validation')),
    target_content_band INTEGER NOT NULL CHECK(target_content_band BETWEEN 1 AND 5),
    target_organization_band INTEGER NOT NULL CHECK(target_organization_band BETWEEN 1 AND 5),
    target_expression_band INTEGER NOT NULL CHECK(target_expression_band BETWEEN 1 AND 5),
    topic_prompt TEXT NOT NULL,
    essay TEXT NOT NULL,
    api_rationale_json TEXT NOT NULL,
    model_rationale_json TEXT NOT NULL,
    first_source TEXT NOT NULL CHECK(first_source IN ('api', 'model'))
);

CREATE TABLE IF NOT EXISTS responses (
    user_name TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    content_score INTEGER CHECK(content_score BETWEEN 1 AND 5),
    organization_score INTEGER CHECK(organization_score BETWEEN 1 AND 5),
    expression_score INTEGER CHECK(expression_score BETWEEN 1 AND 5),
    content_reason TEXT,
    organization_reason TEXT,
    expression_reason TEXT,
    score_submitted_at TEXT,
    rationale_a_source TEXT CHECK(rationale_a_source IN ('api', 'model')),
    rationale_a_content_verdict TEXT CHECK(rationale_a_content_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_a_organization_verdict TEXT CHECK(rationale_a_organization_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_a_expression_verdict TEXT CHECK(rationale_a_expression_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_a_content_reason TEXT,
    rationale_a_organization_reason TEXT,
    rationale_a_expression_reason TEXT,
    rationale_a_submitted_at TEXT,
    rationale_b_source TEXT CHECK(rationale_b_source IN ('api', 'model')),
    rationale_b_content_verdict TEXT CHECK(rationale_b_content_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_b_organization_verdict TEXT CHECK(rationale_b_organization_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_b_expression_verdict TEXT CHECK(rationale_b_expression_verdict IN ('appropriate', 'partial', 'inappropriate')),
    rationale_b_content_reason TEXT,
    rationale_b_organization_reason TEXT,
    rationale_b_expression_reason TEXT,
    rationale_b_submitted_at TEXT,
    PRIMARY KEY(user_name, item_index),
    FOREIGN KEY(item_index) REFERENCES study_items(item_index)
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    csrf_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS responses_user_progress ON responses(user_name, rationale_b_submitted_at);
