"""
Tennis Elo Edge Engine — Web Dashboard
FastAPI backend for Vercel deployment.
"""
import json
import sys
import os
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from engine import (
    EloEngine, download_sackmann_data, build_ratings,
    fetch_tennis_markets, parse_tennis_matchup,
    generate_signals, find_player,
)

app = FastAPI(title="Tennis Elo Edge Engine")

# Cache engine in module scope (rebuilt per cold start)
_engine = None
_engine_info = {}

def get_engine():
    global _engine, _engine_info
    if _engine is None:
        current_year = date.today().year
        years = list(range(current_year - 3, current_year + 1))
        matches = download_sackmann_data(years, "atp")
        _engine = build_ratings(matches)
        _engine_info = {
            "players": len(_engine.players),
            "matches": len(matches),
            "years": years,
        }
    return _engine

@app.get("/")
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Tennis Elo Edge Engine</h1>")

@app.get("/api/health")
async def health():
    from datetime import datetime, timezone
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/scan")
async def scan(bankroll: float = Query(420.0), min_edge: float = Query(0.04)):
    engine = get_engine()
    events = fetch_tennis_markets()
    markets = [m for m in (parse_tennis_matchup(e) for e in events) if m]
    signals = generate_signals(engine, markets, bankroll, min_edge)
    return {
        "engine": _engine_info,
        "matches_scanned": len(markets),
        "signals": [{
            "player_a": s.player_a,
            "player_b": s.player_b,
            "surface": s.surface,
            "is_slam": s.is_slam,
            "model_a": round(s.model_prob_a, 3),
            "model_b": round(s.model_prob_b, 3),
            "market_a": round(s.pm_price_a, 3),
            "market_b": round(s.pm_price_b, 3),
            "edge_a": round(s.edge_a, 3),
            "edge_b": round(s.edge_b, 3),
            "kelly_a": round(s.kelly_a, 3),
            "kelly_b": round(s.kelly_b, 3),
            "size_a": s.size_a,
            "size_b": s.size_b,
            "signal": s.signal,
            "confidence": s.confidence,
            "elo_a": s.elo_a,
            "elo_b": s.elo_b,
            "bo5_boost": s.bo5_boost,
            "fatigue_a": s.fatigue_a,
            "fatigue_b": s.fatigue_b,
            "reasoning": s.reasoning,
            "liquidity": s.liquidity,
            "spread": s.spread,
        } for s in signals],
    }

@app.get("/api/predict")
async def predict(player_a: str = Query(...), player_b: str = Query(...),
                  surface: str = Query("Hard"), slam: bool = Query(False)):
    engine = get_engine()
    a = find_player(engine, player_a)
    b = find_player(engine, player_b)
    if not a:
        return JSONResponse({"error": f"Player not found: {player_a}"}, 404)
    if not b:
        return JSONResponse({"error": f"Player not found: {player_b}"}, 404)
    pred = engine.predict(a, b, surface, match_date=date.today().isoformat(), is_slam=slam)
    return {"player_a": a, "player_b": b, "surface": surface, "slam": slam, **pred}

@app.get("/api/rankings")
async def rankings(surface: str = Query("overall"), limit: int = Query(30)):
    engine = get_engine()
    players = [(n, p) for n, p in engine.players.items() if p.matches_played >= 50]
    if surface == "overall":
        players.sort(key=lambda x: x[1].overall, reverse=True)
    else:
        players.sort(key=lambda x: x[1].effective_elo(surface), reverse=True)
    return [{
        "rank": i+1,
        "name": n,
        "overall": round(p.overall),
        "hard": round(p.hard),
        "clay": round(p.clay),
        "grass": round(p.grass),
        "matches": p.matches_played,
    } for i, (n, p) in enumerate(players[:limit])]
