#!/usr/bin/env python3
# Requires: pip install kivy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import uuid

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty, ListProperty, ObjectProperty, NumericProperty
from kivy.clock import Clock


# =========================
# Core models and base engine
# =========================

@dataclass
class Player:
    id: str
    name: str
    stats: Dict[str, float] = field(default_factory=lambda: {
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "points": 0,  # win=1, tie=0.5 for RR/Swiss
    })

    def __hash__(self):
        return hash(self.id)


@dataclass
class Result:
    winner_id: Optional[str]  # None for tie
    scores: Optional[Tuple[int, int]] = None  # optional (p1_score, p2_score)


@dataclass
class Match:
    id: str
    round: int
    player1: Optional[Player]
    player2: Optional[Player]
    result: Optional[Result] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_bye(self) -> bool:
        return self.player2 is None

    def set_result(self, result: Result):
        self.result = result


class Tournament:
    def __init__(self, players: List[Player], name: str = "Tournament"):
        self.players: List[Player] = list(players)
        self.name = name
        self.round: int = 0
        self.matches_by_round: Dict[int, List[Match]] = {}
        self.played_pairs: Set[Tuple[str, str]] = set()

    def next_round(self) -> List[Match]:
        self.round += 1
        matches = self._generate_round(self.round)
        self.matches_by_round[self.round] = matches
        return matches

    def record_result(self, match_id: str, result: Result):
        match = self._find_match(match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        if match.result is not None:
            raise ValueError("Result already recorded")
        match.set_result(result)
        self._apply_result(match)

    def standings(self) -> List[Player]:
        return sorted(
            self.players,
            key=lambda p: (p.stats.get("points", 0), p.stats.get("wins", 0), -p.stats.get("losses", 0)),
            reverse=True
        )

    def completed(self) -> bool:
        raise NotImplementedError

    def _generate_round(self, round_number: int) -> List[Match]:
        raise NotImplementedError

    def _apply_result(self, match: Match):
        p1, p2 = match.player1, match.player2
        r = match.result
        if p1 and p2 and r:
            pair_key = tuple(sorted([p1.id, p2.id]))
            self.played_pairs.add(pair_key)

            if r.winner_id is None:
                p1.stats["ties"] += 1
                p2.stats["ties"] += 1
                p1.stats["points"] += 0.5
                p2.stats["points"] += 0.5
            elif r.winner_id == p1.id:
                p1.stats["wins"] += 1
                p2.stats["losses"] += 1
                p1.stats["points"] += 1
            elif r.winner_id == p2.id:
                p2.stats["wins"] += 1
                p1.stats["losses"] += 1
                p2.stats["points"] += 1

    def _find_match(self, match_id: str) -> Optional[Match]:
        for round_matches in self.matches_by_round.values():
            for m in round_matches:
                if m.id == match_id:
                    return m
        return None


# =========================
# Single Elimination
# =========================

class SingleElimination(Tournament):
    def __init__(self, players: List[Player], name: str = "Single Elimination"):
        super().__init__(players, name)
        self.bracket_slots: List[Optional[Player]] = self._seed_with_byes(self.players)
        self.active_slots: List[Optional[Player]] = self.bracket_slots[:]
        self.champion: Optional[Player] = None

    def _seed_with_byes(self, players: List[Player]) -> List[Optional[Player]]:
        n = len(players)
        next_pow2 = 1 << (n - 1).bit_length()
        return players[:] + [None] * (next_pow2 - n)

    def _generate_round(self, round_number: int) -> List[Match]:
        matches: List[Match] = []
        for i in range(0, len(self.active_slots), 2):
            p1 = self.active_slots[i]
            p2 = self.active_slots[i + 1] if i + 1 < len(self.active_slots) else None
            matches.append(Match(id=str(uuid.uuid4()), round=round_number, player1=p1, player2=p2))
        return matches

    def _apply_result(self, match: Match):
        super()._apply_result(match)
        p1, p2 = match.player1, match.player2
        r = match.result
        winner = None
        if match.is_bye() and p1:
            winner = p1
        elif r and r.winner_id:
            winner = p1 if r.winner_id == p1.id else p2
        if winner:
            match.metadata["winner_id"] = winner.id

    def completed(self) -> bool:
        if self.champion:
            return True
        current = self.matches_by_round.get(self.round, [])
        if not current:
            return False
        all_recorded = all(m.result is not None or m.is_bye() for m in current)
        if all_recorded:
            winners: List[Optional[Player]] = []
            for m in current:
                wid = m.metadata.get("winner_id")
                if wid is None and m.is_bye() and m.player1:
                    wid = m.player1.id
                winners.append(next((p for p in self.players if p.id == wid), None))
            self.active_slots = winners
            alive = [w for w in self.active_slots if w is not None]
            if len(alive) == 1:
                self.champion = alive[0]
                return True
        return False


# =========================
# Double Elimination (minimal)
# =========================

class DoubleElimination(Tournament):
    def __init__(self, players: List[Player], name: str = "Double Elimination"):
        super().__init__(players, name)
        self.wb_slots: List[Optional[Player]] = self._seed_with_byes(self.players)
        self.lb_slots: List[Optional[Player]] = []
        self.finalists: List[Player] = []
        self.champion: Optional[Player] = None
        self.grand_final_reset_needed: bool = False

    def _seed_with_byes(self, players: List[Player]) -> List[Optional[Player]]:
        n = len(players)
        next_pow2 = 1 << (n - 1).bit_length()
        return players[:] + [None] * (next_pow2 - n)

    def _pair(self, slots: List[Optional[Player]], round_number: int, label: str) -> List[Match]:
        matches = []
        for i in range(0, len(slots), 2):
            p1 = slots[i]
            p2 = slots[i + 1] if i + 1 < len(slots) else None
            if p1 is None and p2 is None:
                continue
            m = Match(id=str(uuid.uuid4()), round=round_number, player1=p1, player2=p2, metadata={"bracket": label})
            matches.append(m)
        return matches

    def _generate_round(self, round_number: int) -> List[Match]:
        matches = []
        if any(p is not None for p in self.wb_slots):
            matches += self._pair(self.wb_slots, round_number, "WB")
        if len(self.lb_slots) > 1 and any(p is not None for p in self.lb_slots):
            matches += self._pair(self.lb_slots, round_number, "LB")
        if len(self.finalists) == 2 and not any(m.metadata.get("type") == "GF" for m in matches):
            matches.append(Match(
                id=str(uuid.uuid4()),
                round=round_number,
                player1=self.finalists[0],
                player2=self.finalists[1],
                metadata={"type": "GF", "bracket": "GF"}
            ))
        return matches

    def _apply_result(self, match: Match):
        super()._apply_result(match)
        r = match.result
        if match.is_bye():
            winner = match.player1
            loser = None
        else:
            if r is None:
                return
            winner = match.player1 if r.winner_id == match.player1.id else match.player2
            loser = match.player2 if winner is match.player1 else match.player1

        bracket = match.metadata.get("bracket")

        if match.metadata.get("type") == "GF":
            wb_player = self.finalists[0]
            if winner == wb_player:
                self.champion = winner
            else:
                self.grand_final_reset_needed = True
            return

        if bracket == "WB":
            self._advance_in_bracket(self.wb_slots, match.player1, match.player2, winner)
            if loser:
                self.lb_slots.append(loser)
        elif bracket == "LB":
            self._advance_in_bracket(self.lb_slots, match.player1, match.player2, winner)

        wb_alive = [p for p in self.wb_slots if p is not None]
        lb_alive = [p for p in self.lb_slots if p is not None]
        if len(wb_alive) == 1 and len(lb_alive) == 1:
            self.finalists = [wb_alive[0], lb_alive[0]]

    def _advance_in_bracket(self, slots: List[Optional[Player]], p1: Optional[Player], p2: Optional[Player], winner: Optional[Player]):
        for idx, p in enumerate(slots):
            if p in (p1, p2):
                slots[idx] = None
        if winner:
            slots.append(winner)
        slots[:] = [p for p in slots if p is not None]

    def completed(self) -> bool:
        if self.champion:
            return True
        if self.grand_final_reset_needed:
            self.round += 1
            gf = Match(
                id=str(uuid.uuid4()),
                round=self.round,
                player1=self.finalists[0],
                player2=self.finalists[1],
                metadata={"type": "GF", "bracket": "GF", "reset": "true"}
            )
            self.matches_by_round.setdefault(self.round, []).append(gf)
            self.grand_final_reset_needed = False
            return False
        return False


# =========================
# Round Robin
# =========================

class RoundRobin(Tournament):
    def __init__(self, players: List[Player], name: str = "Round Robin"):
        super().__init__(players, name)
        self.schedule: List[List[Tuple[Optional[Player], Optional[Player]]]] = self._circle_method(self.players)

    def _circle_method(self, players: List[Player]) -> List[List[Tuple[Optional[Player], Optional[Player]]]]:
        ps = players[:]
        bye: Optional[Player] = None
        if len(ps) % 2 == 1:
            bye = Player(id="BYE", name="BYE")
            ps.append(bye)

        n = len(ps)
        rounds = []
        for _ in range(n - 1):
            pairs = []
            for i in range(n // 2):
                p1 = ps[i]
                p2 = ps[n - 1 - i]
                if bye in (p1, p2):
                    real = p1 if p2 == bye else p2
                    pairs.append((real, None))
                else:
                    pairs.append((p1, p2))
            rounds.append(pairs)
            ps = [ps[0]] + [ps[-1]] + ps[1:-1]
        return rounds

    def _generate_round(self, round_number: int) -> List[Match]:
        if round_number - 1 >= len(self.schedule):
            return []
        matches = []
        for p1, p2 in self.schedule[round_number - 1]:
            matches.append(Match(id=str(uuid.uuid4()), round=round_number, player1=p1, player2=p2))
        return matches

    def _apply_result(self, match: Match):
        super()._apply_result(match)
        if match.is_bye() and match.player1:
            match.player1.stats["wins"] += 1
            match.player1.stats["points"] += 1

    def completed(self) -> bool:
        expected = sum(len(r) for r in self.schedule)
        recorded = 0
        for ms in self.matches_by_round.values():
            for m in ms:
                if m.is_bye() or m.result is not None:
                    recorded += 1
        return recorded >= expected


# =========================
# Swiss
# =========================

class Swiss(Tournament):
    def __init__(self, players: List[Player], name: str = "Swiss"):
        super().__init__(players, name)
        self.byes_given: Dict[str, int] = {p.id: 0 for p in self.players}
        self.max_rounds: Optional[int] = 3

    def _score_key(self, p: Player) -> float:
        return p.stats.get("points", 0)

    def _generate_round(self, round_number: int) -> List[Match]:
        groups: Dict[float, List[Player]] = {}
        for p in self.players:
            groups.setdefault(self._score_key(p), []).append(p)
        ordered = sorted(groups.items(), key=lambda kv: kv[0], reverse=True)

        matches: List[Match] = []
        unpaired: List[Player] = []
        for _, group in ordered:
            group = group[:]
            used = set()
            for i, p in enumerate(group):
                if p.id in used:
                    continue
                partner = None
                for q in group[i+1:]:
                    if q.id in used:
                        continue
                    pair_key = tuple(sorted([p.id, q.id]))
                    if pair_key not in self.played_pairs:
                        partner = q
                        break
                if partner is None:
                    unpaired.append(p)
                    used.add(p.id)
                else:
                    used.add(p.id)
                    used.add(partner.id)
                    matches.append(Match(id=str(uuid.uuid4()), round=round_number, player1=p, player2=partner))
            for q in group:
                if q.id not in used:
                    unpaired.append(q)

        unpaired = sorted(unpaired, key=self._score_key, reverse=True)
        i = 0
        while i < len(unpaired) - 1:
            p = unpaired[i]
            partner_idx = None
            for j in range(i+1, len(unpaired)):
                pair_key = tuple(sorted([p.id, unpaired[j].id]))
                if pair_key not in self.played_pairs:
                    partner_idx = j
                    break
            if partner_idx is None:
                i += 1
            else:
                q = unpaired.pop(partner_idx)
                matches.append(Match(id=str(uuid.uuid4()), round=round_number, player1=p, player2=q))
                unpaired.pop(i)

        if len(unpaired) == 1:
            bye_player = self._pick_bye(unpaired[0], unpaired)
            matches.append(Match(id=str(uuid.uuid4()), round=round_number, player1=bye_player, player2=None))
        return matches

    def _pick_bye(self, candidate: Player, pool: List[Player]) -> Player:
        ranked = sorted(pool + [candidate], key=lambda p: (p.stats.get("points", 0), self.byes_given[p.id]))
        chosen = ranked[0]
        self.byes_given[chosen.id] += 1
        return chosen

    def _apply_result(self, match: Match):
        super()._apply_result(match)
        if match.is_bye() and match.player1:
            match.player1.stats["wins"] += 1
            match.player1.stats["points"] += 1

    def completed(self) -> bool:
        if self.max_rounds is None:
            return False
        played_rounds = len(self.matches_by_round)
        if played_rounds < self.max_rounds:
            return False
        for ms in self.matches_by_round.values():
            for m in ms:
                if m.player2 is not None and m.result is None:
                    return False
        return True


# =========================
# Kivy GUI
# =========================

class MatchesView(RecycleView):
    items = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viewclass = 'Label'
        self.refresh()

    def refresh(self):
        self.data = [{'text': item} for item in self.items]


class StandingsView(RecycleView):
    rows = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viewclass = 'Label'
        self.refresh()

    def refresh(self):
        # render as single column lines
        self.data = [{'text': row} for row in self.rows]


class TournamentRoot(BoxLayout):
    tournament: Optional[Tournament] = None
    players: List[Player] = []
    round_label = ObjectProperty(None)
    players_list = ObjectProperty(None)
    add_input = ObjectProperty(None)
    format_spinner = ObjectProperty(None)
    swiss_spinner = ObjectProperty(None)
    matches_view = ObjectProperty(None)
    standings_view = ObjectProperty(None)
    record_btn = ObjectProperty(None)
    advance_btn = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = BoxLayout(orientation='horizontal', size_hint_y=None, height=120, spacing=10, padding=10)
        self.add_widget(top)

        # Players column
        players_col = BoxLayout(orientation='vertical', size_hint_x=0.5, spacing=6)
        top.add_widget(players_col)

        add_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=36, spacing=6)
        self.add_input = TextInput(hint_text="Player name")
        add_btn = Button(text="Add", size_hint_x=None, width=100, on_release=lambda *_: self.add_player())
        rem_btn = Button(text="Remove selected", size_hint_x=None, width=160, on_release=lambda *_: self.remove_selected_player())
        add_row.add_widget(self.add_input)
        add_row.add_widget(add_btn)
        add_row.add_widget(rem_btn)
        players_col.add_widget(add_row)

        self.players_list = RecycleView(size_hint_y=1)
        self.players_list.viewclass = 'Label'
        self.players_list.data = []
        players_col.add_widget(self.players_list)

        # Settings column
        settings_col = GridLayout(cols=2, size_hint_x=0.5, spacing=6)
        top.add_widget(settings_col)

        settings_col.add_widget(Label(text="Format:", size_hint_y=None, height=28))
        self.format_spinner = Spinner(text="Single Elimination",
                                      values=["Single Elimination", "Double Elimination", "Round Robin", "Swiss"],
                                      size_hint_y=None, height=28)
        settings_col.add_widget(self.format_spinner)

        settings_col.add_widget(Label(text="Swiss rounds:", size_hint_y=None, height=28))
        self.swiss_spinner = Spinner(text="3", values=[str(i) for i in range(1, 11)], size_hint_y=None, height=28)
        settings_col.add_widget(self.swiss_spinner)

        start_btn = Button(text="Start tournament", size_hint_y=None, height=40)
        start_btn.bind(on_release=lambda *_: self.start_tournament())
        settings_col.add_widget(start_btn)
        self.round_label = Label(text="Round: -", size_hint_y=None, height=40)
        settings_col.add_widget(self.round_label)

        # Middle area: matches and standings
        middle = BoxLayout(orientation='horizontal', spacing=10, padding=10)
        self.add_widget(middle)

        # Matches panel
        left = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=6)
        middle.add_widget(left)

        left.add_widget(Label(text="Matches", size_hint_y=None, height=24))
        self.matches_view = MatchesView()
        left.add_widget(self.matches_view)

        btn_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=48, spacing=6)
        self.record_btn = Button(text="Record result", disabled=True)
        self.record_btn.bind(on_release=lambda *_: self.open_record_popup())
        self.advance_btn = Button(text="Advance round", disabled=True)
        self.advance_btn.bind(on_release=lambda *_: self.advance_round())
        btn_row.add_widget(self.record_btn)
        btn_row.add_widget(self.advance_btn)
        left.add_widget(btn_row)

        # Standings panel
        right = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=6)
        middle.add_widget(right)

        right.add_widget(Label(text="Standings", size_hint_y=None, height=24))
        self.standings_view = StandingsView()
        right.add_widget(self.standings_view)

        # Selection helper (simulate selection by tapping a match label)
        self.selected_match_index: Optional[int] = None
        # Bind touch on matches_view to detect index by y position
        self.matches_view.bind(on_touch_down=self._on_touch_matches)

    # ---------- Player management ----------

    def add_player(self):
        name = self.add_input.text.strip()
        if not name:
            return
        player = Player(id=str(uuid.uuid4()), name=name)
        self.players.append(player)
        self.add_input.text = ""
        self._refresh_players()

    def remove_selected_player(self):
        # Remove last added player for simplicity (or enhance with proper selection later)
        if not self.players:
            return
        self.players.pop()
        self._refresh_players()

    def _refresh_players(self):
        self.players_list.data = [{'text': p.name} for p in self.players]

    # ---------- Tournament lifecycle ----------

    def start_tournament(self):
        if len(self.players) < 2:
            self._toast("Need at least 2 players to start.")
            return

        fmt = self.format_spinner.text
        if fmt == "Single Elimination":
            self.tournament = SingleElimination(self.players, "Single Elimination")
        elif fmt == "Double Elimination":
            self.tournament = DoubleElimination(self.players, "Double Elimination")
        elif fmt == "Round Robin":
            self.tournament = RoundRobin(self.players, "Round Robin")
        elif fmt == "Swiss":
            self.tournament = Swiss(self.players, "Swiss")
            try:
                self.tournament.max_rounds = int(self.swiss_spinner.text)
            except ValueError:
                self.tournament.max_rounds = 3
        else:
            self._toast("Unknown format")
            return

        self._load_next_round()

    def _load_next_round(self):
        matches = self.tournament.next_round()
        self.round_label.text = f"Round: {self.tournament.round}"
        items = []
        for m in matches:
            if m.is_bye():
                # Auto-record bye win
                self.tournament.record_result(m.id, Result(winner_id=m.player1.id))
            else:
                items.append(f"{m.player1.name} vs {m.player2.name}  (Round {m.round})")
        self.matches_view.items = items
        self.matches_view.refresh()
        self.selected_match_index = None
        self.record_btn.disabled = True
        self._refresh_standings()
        self._update_controls()

    def _on_touch_matches(self, rv, touch):
        if not self.matches_view.collide_point(*touch.pos):
            return False
        # Determine item index from y position (approximate)
        # Each label has default height ~ (dp) but can vary; we can map via children order
        # Simpler: pick nearest by cycling through data length using relative y
        if not self.matches_view.items:
            return False
        # Use ScrollView offset and container height to estimate index
        container = self.matches_view.children[0] if self.matches_view.children else None
        if not container:
            return False
        # Heuristic: map local y to index from top
        local_y = touch.y - self.matches_view.y
        # Assume ~30px per row; clamp
        row_h = 30.0
        idx_from_bottom = int(local_y // row_h)
        idx = len(self.matches_view.items) - 1 - idx_from_bottom
        if 0 <= idx < len(self.matches_view.items):
            self.selected_match_index = idx
            self.record_btn.disabled = False
        return True

    def open_record_popup(self):
        idx = self.selected_match_index
        if idx is None:
            return
        match = self.tournament.matches_by_round[self.tournament.round][idx]
        if match.result is not None or match.is_bye():
            return

        content = BoxLayout(orientation='vertical', spacing=8, padding=10)
        content.add_widget(Label(text=f"Round {match.round}"))
        content.add_widget(Label(text=f"{match.player1.name} vs {match.player2.name}"))

        choice_row = BoxLayout(orientation='vertical', spacing=6)
        tb1 = ToggleButton(text=f"{match.player1.name} wins", group="win", state="down")
        tb2 = ToggleButton(text=f"{match.player2.name} wins", group="win")
        tb3 = ToggleButton(text="Tie", group="win")
        choice_row.add_widget(tb1)
        choice_row.add_widget(tb2)
        choice_row.add_widget(tb3)
        content.add_widget(choice_row)

        scores_row = GridLayout(cols=2, spacing=6, size_hint_y=None, height=80)
        s1 = TextInput(hint_text=f"{match.player1.name} score", multiline=False)
        s2 = TextInput(hint_text=f"{match.player2.name} score", multiline=False)
        scores_row.add_widget(s1)
        scores_row.add_widget(s2)
        content.add_widget(Label(text="Scores (optional)"))
        content.add_widget(scores_row)

        btns = BoxLayout(orientation='horizontal', spacing=8, size_hint_y=None, height=40)
        popup = Popup(title="Record result", content=content, size_hint=(0.8, 0.6), auto_dismiss=False)

        def save_result(_btn):
            winval = None
            if tb3.state == "down":
                winval = None
            elif tb1.state == "down":
                winval = match.player1.id
            elif tb2.state == "down":
                winval = match.player2.id

            score_tuple = None
            try:
                if s1.text.strip() and s2.text.strip():
                    score_tuple = (int(s1.text.strip()), int(s2.text.strip()))
            except ValueError:
                self._toast("Scores must be integers.")
                return

            try:
                self.tournament.record_result(match.id, Result(winner_id=winval, scores=score_tuple))
            except ValueError as e:
                self._toast(str(e))
                return

            # Update label text
            if winval is None:
                self.matches_view.items[idx] = f"{match.player1.name} drew {match.player2.name}"
            else:
                winner_name = match.player1.name if winval == match.player1.id else match.player2.name
                loser_name = match.player2.name if winval == match.player1.id else match.player1.name
                self.matches_view.items[idx] = f"{winner_name} def. {loser_name}"

            self.matches_view.refresh()
            popup.dismiss()
            self._refresh_standings()
            self._update_controls()
            self._check_completion()

        btns.add_widget(Button(text="Save", on_release=save_result))
        btns.add_widget(Button(text="Cancel", on_release=lambda *_: popup.dismiss()))
        content.add_widget(btns)

        popup.open()

    def advance_round(self):
        if self.tournament.completed():
            champ = getattr(self.tournament, "champion", None)
            if champ:
                self._toast(f"{champ.name} wins!")
            else:
                self._toast("Tournament complete.")
            self.advance_btn.disabled = True
            self.record_btn.disabled = True
            return
        self._load_next_round()

    def _update_controls(self):
        current = self.tournament.matches_by_round.get(self.tournament.round, [])
        all_recorded = all(m.result is not None or m.is_bye() for m in current)
        self.advance_btn.disabled = not all_recorded

    def _check_completion(self):
        if self.tournament.completed():
            champ = getattr(self.tournament, "champion", None)
            if champ:
                self._toast(f"{champ.name} wins!")
            else:
                self._toast("Tournament complete.")
            self.advance_btn.disabled = True

    def _refresh_standings(self):
        lines = []
        for p in self.tournament.standings():
            lines.append(f"{p.name} | W:{int(p.stats.get('wins',0))} "
                         f"L:{int(p.stats.get('losses',0))} T:{int(p.stats.get('ties',0))} "
                         f"P:{p.stats.get('points',0)}")
        self.standings_view.rows = lines
        self.standings_view.refresh()

    def _toast(self, msg: str):
        popup = Popup(title="", content=Label(text=msg), size_hint=(0.5, 0.25))
        popup.open()
        Clock.schedule_once(lambda *_: popup.dismiss(), 1.5)


class TournamentApp(App):
    def build(self):
        return TournamentRoot()


if __name__ == "__main__":
    TournamentApp().run()
