#include <utility>
#include <algorithm>
#include <vector>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * Piece values for MVV-LVA move ordering
 * Index matches piece encoding: 1=P,2=R,3=N,4=B,5=Q,6=K
 *============================================================*/
static constexpr int PIECE_VALUE[7] = { 0, 1, 5, 3, 3, 9, 100 };

/* Returns true if the opponent has a piece on the move's destination. */
static bool is_capture(State* state, const Move& move){
    int opponent = 1 - state->player;
    auto [from, to] = move;
    auto [tr, tc]   = to;
    int bh = state->board_h();
    if((int)tr >= bh) tr = tr % bh;   // promotion encoding: actual row is to_row % bh
    return state->piece_at(opponent, (int)tr, (int)tc) != 0;
}

/*
 * MVV-LVA score: higher = search earlier.
 * Captures: victim_value * 10 - attacker_value  (big victim, small attacker first)
 * Quiet moves: score 0 (searched after all captures)
 */
static int move_score(State* state, const Move& move){
    if(!is_capture(state, move)) return 0;

    int opponent = 1 - state->player;
    auto [from, to] = move;
    auto [fr, fc]   = from;
    auto [tr, tc]   = to;
    int bh = state->board_h();
    if((int)tr >= bh) tr = tr % bh;

    int victim   = state->piece_at(opponent,          (int)tr, (int)tc);
    int attacker = state->piece_at(state->player, (int)fr, (int)fc);

    int v_val = (victim   >= 1 && victim   <= 6) ? PIECE_VALUE[victim]   : 0;
    int a_val = (attacker >= 1 && attacker <= 6) ? PIECE_VALUE[attacker] : 0;

    return v_val * 10 - a_val + 1;   // +1 keeps captures above quiet moves (score 0)
}

/* Sort a move list in-place: captures (MVV-LVA) first, quiet moves last. */
static void order_moves(State* state, std::vector<Move>& moves){
    std::stable_sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_score(state, a) > move_score(state, b);
        }
    );
}


/*============================================================
 * MiniMax — quiescence_search
 *
 * Search only capture moves at the horizon to avoid the
 * horizon effect.  Uses a stand-pat score as a lower bound:
 * if the static eval is already >= beta we can cut off.
 *============================================================*/
int MiniMax::quiescence_search(
    State *state,
    int qdepth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    /* === Stand-pat: static eval as a lower bound === */
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);

    if(stand_pat >= beta) return stand_pat;
    if(stand_pat > alpha) alpha = stand_pat;

    /* === Depth exhausted: position is quiet enough === */
    if(qdepth <= 0) return stand_pat;

    /* === Search captures only, MVV-LVA ordered === */
    int best_score = stand_pat;

    // Build a local capture-only move list so we can sort without touching state
    std::vector<Move> captures;
    captures.reserve(state->legal_actions.size());
    for(auto& action : state->legal_actions){
        if(is_capture(state, action)) captures.push_back(action);
    }
    if(p.use_move_ordering) order_moves(state, captures);

    for(auto& action : captures){
        State* next  = state->next_state(action);
        bool   same  = next->same_player_as_parent();

        int score;
        if(same){
            score =  quiescence_search(next, qdepth-1, history, ply+1, ctx, p,  alpha,  beta);
        } else {
            score = -quiescence_search(next, qdepth-1, history, ply+1, ctx, p, -beta,  -alpha);
        }
        delete next;

        if(score > best_score) best_score = score;
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta)      break;
    }

    return best_score;
}


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with alpha-beta, optional PVS, quiescence, and
 * move ordering. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    history.push(state->hash());

    if(depth <= 0){
        int score;
        if(p.use_quiescence){
            score = quiescence_search(
                state, p.quiescence_max_depth, history, ply, ctx, p, alpha, beta
            );
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(state->hash());
        return score;
    }

    /* === Move ordering === */
    std::vector<Move> moves = state->legal_actions;
    if(p.use_move_ordering) order_moves(state, moves);

    /* === Negamax / PVS loop === */
    int  best_score  = M_MAX;
    bool first_child = true;

    for(auto& action : moves){
        State* next = state->next_state(action);
        bool   same = next->same_player_as_parent();

        int score;
        if(same){
            // same player: do not swap signs
            score = eval_ctx(next, depth-1, history, ply+1, ctx, p, alpha, beta);
        } else if(!p.use_pvs || first_child){
            // plain negamax OR PVS first child: full window
            score = -eval_ctx(next, depth-1, history, ply+1, ctx, p, -beta, -alpha);
        } else {
            // PVS: null-window search for subsequent children
            score = -eval_ctx(next, depth-1, history, ply+1, ctx, p, -(alpha+1), -alpha);
            // Re-search with full window if it failed high
            if(score > alpha && score < beta){
                score = -eval_ctx(next, depth-1, history, ply+1, ctx, p, -beta, -alpha);
            }
        }

        delete next;

        if(score > best_score) best_score = score;
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta)      break;   // beta cutoff

        first_child = false;
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    /* === Move ordering at root === */
    std::vector<Move> moves = state->legal_actions;
    if(p.use_move_ordering) order_moves(state, moves);

    int best_score  = M_MAX - 10;
    int move_index  = 0;
    int total_moves = (int)moves.size();
    int alpha       = M_MAX;
    int beta        = P_MAX;
    bool first_child = true;

    for(auto& action : moves){
        State* next = state->next_state(action);
        bool   same = next->same_player_as_parent();
        int score;

        if(same){
            score = eval_ctx(next, depth-1, history, 1, ctx, p, alpha, beta);
        } else if(!p.use_pvs || first_child){
            // plain negamax OR PVS first child: full window
            score = -eval_ctx(next, depth-1, history, 1, ctx, p, -beta, -alpha);
        } else {
            // PVS: null-window search for subsequent children
            score = -eval_ctx(next, depth-1, history, 1, ctx, p, -(alpha+1), -alpha);
            if(score > alpha && score < beta){
                score = -eval_ctx(next, depth-1, history, 1, ctx, p, -beta, -alpha);
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta)      break;   // beta cutoff at root

        first_child = false;
        move_index++;
    }

    result.score = best_score;
    if(p.report_partial && ctx.on_root_update){
        ctx.on_root_update({result.best_move, best_score, depth, total_moves, total_moves});
    }
    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval",          "true"},
        {"UseEvalMobility",    "true"},
        {"UsePVS",             "true"},
        {"UseQuiescence",      "true"},
        {"QuiescenceMaxDepth", "16"},
        {"UseMoveOrdering",    "true"},
        {"ReportPartial",      "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval",          ParamDef::CHECK, "true"},
        {"UseEvalMobility",    ParamDef::CHECK, "true"},
        {"UsePVS",             ParamDef::CHECK, "true"},
        {"UseQuiescence",      ParamDef::CHECK, "true"},
        {"QuiescenceMaxDepth", ParamDef::SPIN,  "16", 1, 64},
        {"UseMoveOrdering",    ParamDef::CHECK, "true"},
        {"ReportPartial",      ParamDef::CHECK, "true"},
    };
}
