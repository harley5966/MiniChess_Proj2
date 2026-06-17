#include <utility>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with pruning. Caller manages memory.
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
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    if(state->game_state == WIN){
        return P_MAX - ply; //ply: the smaller, the better
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    /* === Negamax / PVS loop === */
    int best_score = M_MAX;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(same){
            // same player: do not swap signs
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
        } else if(!p.use_pvs || first_child){
            // plain negamax, or PVS first child: full window
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        } else {
            // PVS: null-window search for subsequent children
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -(alpha + 1), -alpha);
            // re-search with full window if failed high
            if(score > alpha && score < beta){
                score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }

        if(best_score > alpha){
            alpha = best_score;
        }

        if(alpha >= beta){
            // beta cutoff
            break;
        }

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


    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    int alpha = M_MAX;
    int beta = P_MAX;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        if(same){
            score = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, beta);
        } else if(!p.use_pvs || first_child){
            // plain negamax, or PVS first child: full window
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
        } else {
            // PVS: null-window search for subsequent children
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -(alpha + 1), -alpha);
            // re-search with full window if failed high
            if(score > alpha && score < beta){
                score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
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

        if(best_score > alpha){
            alpha = best_score;
        }
        if(alpha >= beta){
            break;
        }

        first_child = false;
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
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
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"UsePVS", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
