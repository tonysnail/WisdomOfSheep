from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .interest import InterestRecord


class PostListItem(BaseModel):
    post_id: str
    title: str
    platform: str
    source: str
    url: str
    scraped_at: Optional[str] = None
    posted_at: Optional[str] = None
    preview: str
    markets: List[str] = Field(default_factory=list)
    signal: Dict[str, Any] = Field(default_factory=dict)
    has_summary: bool = False
    has_analysis: bool = False
    summary_bullets: List[str] = Field(default_factory=list)
    assets_mentioned: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    spam_likelihood_pct: int = 0
    spam_why: str = ""
    interest: Optional[InterestRecord] = None
    chairman_plain_english: Optional[str] = None
    chairman_direction: Optional[str] = None


class CalendarDay(BaseModel):
    date: str
    count: int
    analysed_count: int = 0


class PostsCalendarResponse(BaseModel):
    days: List[CalendarDay] = Field(default_factory=list)


class PostRow(BaseModel):
    post_id: str
    platform: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    scraped_at: Optional[str] = None
    posted_at: Optional[str] = None
    score: Optional[int] = None
    text: Optional[str] = None


class PostDetail(BaseModel):
    post: PostRow
    extras: Dict[str, Any] = Field(default_factory=dict)
    stages: Dict[str, Any] = Field(default_factory=dict)
    interest: Optional[InterestRecord] = None


class ResearchTickerPayload(BaseModel):
    ticker: str
    article_time: str
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    plan: Dict[str, Any] = Field(default_factory=dict)
    technical: Dict[str, Any] = Field(default_factory=dict)
    sentiment: Dict[str, Any] = Field(default_factory=dict)
    summary_text: str = ""
    updated_at: str
    session_id: Optional[str] = None
    log: str = ""


class ResearchPayload(BaseModel):
    article_time: str
    updated_at: str
    ordered_tickers: List[str] = Field(default_factory=list)
    tickers: Dict[str, ResearchTickerPayload] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    ok: bool
    research: ResearchPayload


class RunStageRequest(BaseModel):
    post_id: str
    stages: List[str]
    overwrite: bool
    refresh_from_csv: bool = False
    echo_post: bool = False


class RunStageResponse(BaseModel):
    ok: bool
    post_id: str


class BatchRunFilter(BaseModel):
    query: Optional[str] = None
    platform: Optional[str] = None
    source: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class BatchRunRequest(BaseModel):
    stages: List[str]
    overwrite: bool
    post_ids: Optional[List[str]] = None
    filter: Optional[BatchRunFilter] = None
    refresh_from_csv: bool = False


class BatchRunResponse(BaseModel):
    ok: bool
    submitted: int
    results: List[RunStageResponse]


__all__ = [
    "BatchRunFilter",
    "BatchRunRequest",
    "BatchRunResponse",
    "CalendarDay",
    "PostDetail",
    "PostListItem",
    "PostRow",
    "PostsCalendarResponse",
    "ResearchPayload",
    "ResearchResponse",
    "ResearchTickerPayload",
    "RunStageRequest",
    "RunStageResponse",
]
