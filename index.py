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
async def predict(
    player_a: str = Query(...), player_b: str = Query(...),
    surface: str = Query("Hard"), slam: bool = Query(False),
    market_price_a: float = Query(None, description="Polymarket price for player A (0-1)"),
    bankroll: float = Query(420.0),
):
    """
    Predict matchup with optional Kelly sizing against a market price.
    If you supply market_price_a, you get edge + Kelly + Manski check.
    """
    engine = get_engine()
    a = find_player(engine, player_a)
    b = find_player(engine, player_b)
    if not a:
        return JSONResponse({"error": f"Player not found: {player_a}. Try last name only."}, 404)
    if not b:
        return JSONResponse({"error": f"Player not found: {player_b}. Try last name only."}, 404)
    pred = engine.predict(a, b, surface, match_date=date.today().isoformat(), is_slam=slam)

    result = {"player_a": a, "player_b": b, "surface": surface, "slam": slam, **pred}

    # If market price supplied, compute edge + Kelly + Manski
    if market_price_a is not None and 0 < market_price_a < 1:
        from engine import compute_kelly
        market_b = 1 - market_price_a
        edge_a = pred["p_a"] - market_price_a
        edge_b = pred["p_b"] - market_b
        kelly_a = compute_kelly(pred["p_a"], market_price_a)
        kelly_b = compute_kelly(pred["p_b"], market_b)
        manski = max(abs(edge_a), abs(edge_b)) > 0.15

        if manski:
            signal = "MANSKI"
            reasoning = f"Edge {max(abs(edge_a), abs(edge_b)):.1%} > 15% -- likely model error [Manski 2006]"
        elif edge_a >= 0.04 and edge_a > edge_b:
            signal = f"BUY {a.split()[-1]}"
            reasoning = f"Model {pred['p_a']:.1%} > Market {market_price_a:.1%}, edge {edge_a:+.1%}"
        elif edge_b >= 0.04 and edge_b > edge_a:
            signal = f"BUY {b.split()[-1]}"
            reasoning = f"Model {pred['p_b']:.1%} > Market {market_b:.1%}, edge {edge_b:+.1%}"
        else:
            signal = "PASS"
            reasoning = f"Max edge {max(edge_a, edge_b):.1%} < 4% threshold"

        result["market"] = {
            "price_a": market_price_a,
            "price_b": round(market_b, 3),
            "edge_a": round(edge_a, 4),
            "edge_b": round(edge_b, 4),
            "kelly_a": round(kelly_a, 4),
            "kelly_b": round(kelly_b, 4),
            "size_a": round(bankroll * min(kelly_a, 0.10), 2),
            "size_b": round(bankroll * min(kelly_b, 0.10), 2),
            "signal": signal,
            "manski_flag": manski,
            "reasoning": reasoning,
        }

    return result


@app.get("/api/ingest")
async def ingest_result(
    winner: str = Query(...), loser: str = Query(...),
    surface: str = Query("Hard"), score: str = Query("6-4 6-3"),
    tourney_level: str = Query("G"), round_name: str = Query("R64"),
):
    """
    Manually ingest a match result to update Elo ratings in real-time.
    This persists until the next Vercel cold start.
    """
    engine = get_engine()
    w = find_player(engine, winner)
    l = find_player(engine, loser)
    if not w:
        # Create new player entry
        w = winner
    if not l:
        l = loser

    from engine import parse_score
    w_sets, l_sets, w_games, l_games = parse_score(score)

    engine.update(
        winner=w, loser=l, surface=surface,
        tourney_level=tourney_level, round_name=round_name,
        w_sets=w_sets, l_sets=l_sets,
        w_games=w_games, l_games=l_games,
        match_date=date.today().isoformat(),
    )

    pw = engine.get_player(w)
    pl = engine.get_player(l)

    return {
        "status": "updated",
        "winner": w,
        "loser": l,
        "score": score,
        "surface": surface,
        "winner_elo": {
            "overall": round(pw.overall),
            "surface": round(pw.surface_elo(surface)),
            "effective": round(pw.effective_elo(surface)),
        },
        "loser_elo": {
            "overall": round(pl.overall),
            "surface": round(pl.surface_elo(surface)),
            "effective": round(pl.effective_elo(surface)),
        },
    }


@app.get("/api/search")
async def search_players(q: str = Query(..., min_length=2)):
    """Search for players by partial name match."""
    engine = get_engine()
    q_lower = q.lower()
    matches = []
    for name, p in engine.players.items():
        if q_lower in name.lower():
            matches.append({
                "name": name,
                "overall": round(p.overall),
                "hard": round(p.hard),
                "clay": round(p.clay),
                "grass": round(p.grass),
                "matches": p.matches_played,
                "last_match": p.last_match_date,
            })
    matches.sort(key=lambda x: x["overall"], reverse=True)
    return matches[:20]


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
