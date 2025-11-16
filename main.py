#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import uuid
import math


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
        for r in range(n - 1):
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
# GUI Application
# =========================

class TournamentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tournament Shrine")
        self.geometry("980x680")
        self.minsize(880, 600)

        self.players: List[Player] = []
        self.tournament: Optional[Tournament] = None

        self._build_layout()

    def _build_layout(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        # Players section
        players_frame = ttk.LabelFrame(top, text="Players")
        players_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.player_name_var = tk.StringVar()
        entry = ttk.Entry(players_frame, textvariable=self.player_name_var)
        entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        players_frame.columnconfigure(0, weight=1)

        add_btn = ttk.Button(players_frame, text="Add", command=self.add_player)
        add_btn.grid(row=0, column=1, padx=5, pady=5)

        remove_btn = ttk.Button(players_frame, text="Remove selected", command=self.remove_selected_player)
        remove_btn.grid(row=0, column=2, padx=5, pady=5)

        self.player_listbox = tk.Listbox(players_frame, height=8)
        self.player_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        players_frame.rowconfigure(1, weight=1)

        # Tournament settings
        settings_frame = ttk.LabelFrame(top, text="Tournament settings")
        settings_frame.pack(side="left", fill="both", expand=True, padx=5)

        ttk.Label(settings_frame, text="Format:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.format_var = tk.StringVar(value="Single Elimination")
        fmt_combo = ttk.Combobox(settings_frame, textvariable=self.format_var, state="readonly",
                                 values=["Single Elimination", "Double Elimination", "Round Robin", "Swiss"])
        fmt_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(settings_frame, text="Swiss rounds:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.swiss_rounds_var = tk.IntVar(value=3)
        swiss_spin = ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.swiss_rounds_var, width=5)
        swiss_spin.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        start_btn = ttk.Button(settings_frame, text="Start tournament", command=self.start_tournament)
        start_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        self.round_label = ttk.Label(settings_frame, text="Round: -")
        self.round_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Main panes
        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Matches pane
        matches_panel = ttk.Frame(main)
        main.add(matches_panel, weight=3)

        matches_frame = ttk.LabelFrame(matches_panel, text="Matches")
        matches_frame.pack(fill="both", expand=True)

        self.matches_list = tk.Listbox(matches_frame)
        self.matches_list.pack(fill="both", expand=True, padx=5, pady=5)

        btns = ttk.Frame(matches_frame)
        btns.pack(fill="x", padx=5, pady=5)
        self.record_btn = ttk.Button(btns, text="Record result", command=self.record_result_dialog, state="disabled")
        self.record_btn.pack(side="left", padx=5)
        self.advance_btn = ttk.Button(btns, text="Advance round", command=self.advance_round, state="disabled")
        self.advance_btn.pack(side="left", padx=5)

        self.matches_list.bind("<<ListboxSelect>>", self._on_match_select)

        # Standings pane
        standings_panel = ttk.Frame(main)
        main.add(standings_panel, weight=2)

        standings_frame = ttk.LabelFrame(standings_panel, text="Standings")
        standings_frame.pack(fill="both", expand=True)

        self.standings = ttk.Treeview(standings_frame, columns=("wins", "losses", "ties", "points"), show="headings")
        self.standings.heading("wins", text="Wins")
        self.standings.heading("losses", text="Losses")
        self.standings.heading("ties", text="Ties")
        self.standings.heading("points", text="Points")
        self.standings.column("wins", width=60, anchor="center")
        self.standings.column("losses", width=60, anchor="center")
        self.standings.column("ties", width=60, anchor="center")
        self.standings.column("points", width=80, anchor="center")
        self.standings.pack(fill="both", expand=True, padx=5, pady=5)

    # ---------- Player management ----------

    def add_player(self):
        name = self.player_name_var.get().strip()
        if not name:
            return
        player = Player(id=str(uuid.uuid4()), name=name)
        self.players.append(player)
        self.player_listbox.insert("end", player.name)
        self.player_name_var.set("")

    def remove_selected_player(self):
        sel = self.player_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.player_listbox.delete(idx)
        self.players.pop(idx)

    # ---------- Tournament lifecycle ----------

    def start_tournament(self):
        if len(self.players) < 2:
            messagebox.showerror("Error", "Need at least 2 players to start.")
            return

        fmt = self.format_var.get()
        if fmt == "Single Elimination":
            self.tournament = SingleElimination(self.players, "Single Elimination")
        elif fmt == "Double Elimination":
            self.tournament = DoubleElimination(self.players, "Double Elimination")
        elif fmt == "Round Robin":
            self.tournament = RoundRobin(self.players, "Round Robin")
        elif fmt == "Swiss":
            self.tournament = Swiss(self.players, "Swiss")
            self.tournament.max_rounds = self.swiss_rounds_var.get()
        else:
            messagebox.showerror("Error", "Unknown format")
            return

        self._load_next_round()

    def _load_next_round(self):
        matches = self.tournament.next_round()
        self.round_label.config(text=f"Round: {self.tournament.round}")
        self.matches_list.delete(0, "end")
        for m in matches:
            if m.is_bye():
                # Auto-record bye win
                self.tournament.record_result(m.id, Result(winner_id=m.player1.id))
            else:
                self.matches_list.insert("end", f"{m.player1.name} vs {m.player2.name}  (Round {m.round})")
        self._refresh_standings()
        self._update_controls()

    def _on_match_select(self, _event):
        # Enable record button if a selectable match exists
        self._update_controls()

    def _update_controls(self):
        has_selection = bool(self.matches_list.curselection())
        self.record_btn.config(state="normal" if has_selection else "disabled")

        # Enable advance when all matches in current round are recorded
        current = self.tournament.matches_by_round.get(self.tournament.round, [])
        all_recorded = all(m.result is not None or m.is_bye() for m in current)
        self.advance_btn.config(state="normal" if all_recorded else "disabled")

    def record_result_dialog(self):
        sel = self.matches_list.curselection()
        if not sel:
            return
        idx = sel[0]
        match = self.tournament.matches_by_round[self.tournament.round][idx]
        if match.result is not None or match.is_bye():
            return

        dialog = tk.Toplevel(self)
        dialog.title("Record result")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text=f"Round {match.round}").grid(row=0, column=0, columnspan=2, padx=8, pady=8)
        ttk.Label(dialog, text=f"{match.player1.name} vs {match.player2.name}").grid(row=1, column=0, columnspan=2, padx=8, pady=8)

        winner_var = tk.StringVar(value=match.player1.id)
        ttk.Radiobutton(dialog, text=f"{match.player1.name} wins", variable=winner_var, value=match.player1.id).grid(row=2, column=0, sticky="w", padx=8)
        ttk.Radiobutton(dialog, text=f"{match.player2.name} wins", variable=winner_var, value=match.player2.id).grid(row=3, column=0, sticky="w", padx=8)
        ttk.Radiobutton(dialog, text="Tie", variable=winner_var, value="TIE").grid(row=4, column=0, sticky="w", padx=8)

        ttk.Label(dialog, text="Scores (optional):").grid(row=5, column=0, padx=8, pady=(10, 2), sticky="w")
        p1_score_var = tk.StringVar()
        p2_score_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=p1_score_var, width=6).grid(row=5, column=1, padx=4, sticky="w")
        ttk.Entry(dialog, textvariable=p2_score_var, width=6).grid(row=6, column=1, padx=4, sticky="w")
        ttk.Label(dialog, text=f"{match.player1.name} score").grid(row=6, column=0, padx=8, sticky="w")
        ttk.Label(dialog, text=f"{match.player2.name} score").grid(row=7, column=0, padx=8, sticky="w")

        btns = ttk.Frame(dialog)
        btns.grid(row=8, column=0, columnspan=2, pady=10)
        def submit():
            winval = winner_var.get()
            score_tuple = None
            try:
                if p1_score_var.get() and p2_score_var.get():
                    score_tuple = (int(p1_score_var.get()), int(p2_score_var.get()))
            except ValueError:
                messagebox.showerror("Error", "Scores must be integers.")
                return

            if winval == "TIE":
                res = Result(winner_id=None, scores=score_tuple)
            else:
                res = Result(winner_id=winval, scores=score_tuple)

            try:
                self.tournament.record_result(match.id, res)
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return

            # Update match line to show result
            if winval == "TIE":
                text = f"{match.player1.name} drew {match.player2.name}"
            else:
                winner_name = match.player1.name if winval == match.player1.id else match.player2.name
                loser_name = match.player2.name if winval == match.player1.id else match.player1.name
                text = f"{winner_name} def. {loser_name}"
            self.matches_list.delete(idx)
            self.matches_list.insert(idx, text)

            dialog.destroy()
            self._refresh_standings()
            self._update_controls()
            self._check_completion()

        ttk.Button(btns, text="Save", command=submit).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

        dialog.wait_window(dialog)

    def advance_round(self):
        # If completed, show champion or completion message
        if self.tournament.completed():
            champ = getattr(self.tournament, "champion", None)
            if champ:
                messagebox.showinfo("Champion", f"{champ.name} wins!")
            else:
                messagebox.showinfo("Complete", "Tournament complete.")
            return
        self._load_next_round()

    def _check_completion(self):
        if self.tournament.completed():
            champ = getattr(self.tournament, "champion", None)
            if champ:
                messagebox.showinfo("Champion", f"{champ.name} wins!")
            else:
                messagebox.showinfo("Complete", "Tournament complete.")
            self.advance_btn.config(state="disabled")
        else:
            current = self.tournament.matches_by_round.get(self.tournament.round, [])
            all_recorded = all(m.result is not None or m.is_bye() for m in current)
            self.advance_btn.config(state="normal" if all_recorded else "disabled")

    def _refresh_standings(self):
        # Only meaningful for RR/Swiss, but we show stats for all formats
        for row in self.standings.get_children():
            self.standings.delete(row)
        for p in self.tournament.standings():
            self.standings.insert("", "end", values=(
                int(p.stats.get("wins", 0)),
                int(p.stats.get("losses", 0)),
                int(p.stats.get("ties", 0)),
                p.stats.get("points", 0),
            ))


def main():
    app = TournamentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
