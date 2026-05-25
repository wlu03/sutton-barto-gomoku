"""FastAPI app for the Gomoku research UI.

Endpoints:
    GET  /                         -> single-page UI
    GET  /api/agents               -> registered agents
    POST /api/games                -> create live game
    GET  /api/games/{gid}          -> game state
    POST /api/games/{gid}/move     -> human move
    POST /api/games/{gid}/agent    -> request agent move
    POST /api/replay               -> validate + expand a move list into plies

In-memory game store. Single-user research tool, not multi-tenant.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gomoku.env import BLACK, GomokuEnv, WHITE
from gomoku.web.registry import autoload, get_factory, list_agents

autoload()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Gomoku")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------- game store ----------


class Game:
    def __init__(self, size: int, black: str, white: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.env = GomokuEnv(size=size)
        self.env.reset()
        self.black = black
        self.white = white
        self.history: list[dict[str, Any]] = []
        self.agents: dict[int, Any] = {}
        for player, slot in ((BLACK, black), (WHITE, white)):
            if slot.startswith("agent:"):
                name = slot.split(":", 1)[1]
                self.agents[player] = get_factory(name)()

    def actor(self) -> str:
        if self.env.done:
            return "none"
        slot = self.black if self.env.to_play == BLACK else self.white
        return "human" if slot == "human" else "agent"

    def state(self) -> dict[str, Any]:
        return {
            "game_id": self.id,
            "size": self.env.size,
            "board": self.env.board.tolist(),
            "to_play": int(self.env.to_play),
            "terminal": self.env.done,
            "winner": int(self.env.winner),
            "last_move": self.history[-1]["action"] if self.history else None,
            "history": self.history,
            "black": self.black,
            "white": self.white,
            "next_actor": self.actor(),
        }


_GAMES: dict[str, Game] = {}


# ---------- request bodies ----------


class NewGame(BaseModel):
    size: int = 15
    black: str = "human"
    white: str = "human"


class HumanMove(BaseModel):
    action: int


class ReplayBody(BaseModel):
    size: int = 15
    moves: list[int]


# ---------- endpoints ----------


@app.get("/api/agents")
def get_agents() -> dict[str, Any]:
    return {"agents": list_agents()}


@app.post("/api/games")
def new_game(body: NewGame) -> dict[str, Any]:
    try:
        g = Game(size=body.size, black=body.black, white=body.white)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    _GAMES[g.id] = g
    return g.state()


@app.get("/api/games/{gid}")
def get_game(gid: str) -> dict[str, Any]:
    g = _GAMES.get(gid)
    if not g:
        raise HTTPException(404, "no such game")
    return g.state()


@app.post("/api/games/{gid}/move")
def human_move(gid: str, body: HumanMove) -> dict[str, Any]:
    g = _require(gid)
    if g.actor() != "human":
        raise HTTPException(409, f"not human's turn (next: {g.actor()})")
    try:
        return _apply(g, body.action, ms=0.0)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/games/{gid}/agent")
def agent_step(gid: str) -> dict[str, Any]:
    g = _require(gid)
    if g.actor() != "agent":
        raise HTTPException(409, f"not agent's turn (next: {g.actor()})")
    agent = g.agents[g.env.to_play]
    t0 = time.perf_counter()
    action = int(agent.act(g.env.board.copy(), int(g.env.to_play), g.env.legal_mask()))
    ms = (time.perf_counter() - t0) * 1000.0
    debug = getattr(agent, "last_debug", None)
    try:
        return _apply(g, action, ms=ms, debug=debug)
    except ValueError as e:
        raise HTTPException(500, f"agent returned illegal move {action}: {e}") from e


@app.post("/api/replay")
def replay(body: ReplayBody) -> dict[str, Any]:
    env = GomokuEnv(size=body.size)
    env.reset()
    plies: list[dict[str, Any]] = [
        {
            "ply": 0,
            "player": None,
            "action": None,
            "board": env.board.tolist(),
            "to_play": int(env.to_play),
            "terminal": False,
            "winner": 0,
        }
    ]
    for i, a in enumerate(body.moves):
        mover = int(env.to_play)
        try:
            s = env.step(int(a))
        except ValueError as e:
            raise HTTPException(400, f"illegal move at ply {i + 1}: {e}") from e
        plies.append(
            {
                "ply": i + 1,
                "player": mover,
                "action": int(a),
                "board": env.board.tolist(),
                "to_play": int(s.to_play),
                "terminal": s.terminal,
                "winner": int(s.winner),
            }
        )
        if s.terminal:
            break
    return {"size": body.size, "plies": plies}


# ---------- helpers ----------


def _require(gid: str) -> Game:
    g = _GAMES.get(gid)
    if not g:
        raise HTTPException(404, "no such game")
    return g


def _apply(g: Game, action: int, ms: float, debug: Any = None) -> dict[str, Any]:
    mover = int(g.env.to_play)
    g.env.step(action)
    g.history.append(
        {
            "ply": len(g.history) + 1,
            "player": mover,
            "action": int(action),
            "ms": round(ms, 2),
            "debug": debug,
        }
    )
    return g.state()
