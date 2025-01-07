icetime_relative_features = [
    "playerId", "name", "season", 'height', 'weight', "age", "position", "icetime",

    # Player Generated Metrics (I_F)
    "I_F_xOnGoal", "I_F_xGoals", "I_F_xRebounds", "I_F_xFreeze", "I_F_xPlayStopped", "I_F_xPlayContinuedInZone",
    "I_F_xPlayContinuedOutsideZone", "I_F_flurryAdjustedxGoals", "I_F_scoreVenueAdjustedxGoals",
    "I_F_flurryScoreVenueAdjustedxGoals", "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_shotsOnGoal",
    "I_F_missedShots", "I_F_blockedShotAttempts", "I_F_shotAttempts", "I_F_points", "I_F_goals", "I_F_rebounds",
    "I_F_reboundGoals", "I_F_freeze", "I_F_playStopped", "I_F_playContinuedInZone", "I_F_playContinuedOutsideZone",
    "I_F_savedShotsOnGoal", "I_F_savedUnblockedShotAttempts", "I_F_penalityMinutes", "I_F_faceOffsWon", "I_F_hits",
    "I_F_takeaways", "I_F_giveaways", "I_F_lowDangerShots", "I_F_mediumDangerShots", "I_F_highDangerShots",
    "I_F_lowDangerxGoals", "I_F_mediumDangerxGoals", "I_F_highDangerxGoals", "I_F_lowDangerGoals", "I_F_mediumDangerGoals",
    "I_F_highDangerGoals", "I_F_scoreAdjustedShotsAttempts", "I_F_unblockedShotAttempts",
    "I_F_scoreAdjustedUnblockedShotAttempts", "I_F_dZoneGiveaways", "I_F_xGoalsFromxReboundsOfShots",
    "I_F_xGoalsFromActualReboundsOfShots", "I_F_reboundxGoals", "I_F_xGoals_with_earned_rebounds",
    "I_F_xGoals_with_earned_rebounds_scoreAdjusted", "I_F_xGoals_with_earned_rebounds_scoreFlurryAdjusted",
    "I_F_shifts", "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts", "I_F_neutralZoneShiftStarts", "I_F_flyShiftStarts",
    "I_F_oZoneShiftEnds", "I_F_dZoneShiftEnds", "I_F_neutralZoneShiftEnds", "I_F_flyShiftEnds",

    # On-Ice For Stats (when player is on ice)
    "OnIce_F_xOnGoal", "OnIce_F_xGoals", "OnIce_F_flurryAdjustedxGoals", "OnIce_F_scoreVenueAdjustedxGoals",
    "OnIce_F_flurryScoreVenueAdjustedxGoals", "OnIce_F_shotsOnGoal", "OnIce_F_missedShots", "OnIce_F_blockedShotAttempts",
    "OnIce_F_shotAttempts", "OnIce_F_goals", "OnIce_F_rebounds", "OnIce_F_reboundGoals", "OnIce_F_lowDangerShots",
    "OnIce_F_mediumDangerShots", "OnIce_F_highDangerShots", "OnIce_F_lowDangerxGoals", "OnIce_F_mediumDangerxGoals",
    "OnIce_F_highDangerxGoals", "OnIce_F_lowDangerGoals", "OnIce_F_mediumDangerGoals", "OnIce_F_highDangerGoals",
    "OnIce_F_scoreAdjustedShotsAttempts", "OnIce_F_unblockedShotAttempts", "OnIce_F_scoreAdjustedUnblockedShotAttempts",
    "OnIce_F_xGoalsFromxReboundsOfShots", "OnIce_F_xGoalsFromActualReboundsOfShots", "OnIce_F_reboundxGoals",
    "OnIce_F_xGoals_with_earned_rebounds", "OnIce_F_xGoals_with_earned_rebounds_scoreAdjusted",
    "OnIce_F_xGoals_with_earned_rebounds_scoreFlurryAdjusted",

    # On-Ice Against Stats (when player is on ice)
    "OnIce_A_xOnGoal", "OnIce_A_xGoals", "OnIce_A_flurryAdjustedxGoals", "OnIce_A_scoreVenueAdjustedxGoals",
    "OnIce_A_flurryScoreVenueAdjustedxGoals", "OnIce_A_shotsOnGoal", "OnIce_A_missedShots", "OnIce_A_blockedShotAttempts",
    "OnIce_A_shotAttempts", "OnIce_A_goals", "OnIce_A_rebounds", "OnIce_A_reboundGoals", "OnIce_A_lowDangerShots",
    "OnIce_A_mediumDangerShots", "OnIce_A_highDangerShots", "OnIce_A_lowDangerxGoals", "OnIce_A_mediumDangerxGoals",
    "OnIce_A_highDangerxGoals", "OnIce_A_lowDangerGoals", "OnIce_A_mediumDangerGoals", "OnIce_A_highDangerGoals",
    "OnIce_A_scoreAdjustedShotsAttempts", "OnIce_A_unblockedShotAttempts", "OnIce_A_scoreAdjustedUnblockedShotAttempts",
    "OnIce_A_xGoalsFromxReboundsOfShots", "OnIce_A_xGoalsFromActualReboundsOfShots", "OnIce_A_reboundxGoals",
    "OnIce_A_xGoals_with_earned_rebounds", "OnIce_A_xGoals_with_earned_rebounds_scoreAdjusted",
    "OnIce_A_xGoals_with_earned_rebounds_scoreFlurryAdjusted",

    # Off-Ice Stats (when player is not on ice)
    "OffIce_F_xGoals", "OffIce_A_xGoals", "OffIce_F_shotAttempts", "OffIce_A_shotAttempts"
]

metrics_no_icetime = [
    "playerId", "season", "name", 'height', 'weight', "age", "bmi", "position",

    # Player Generated Metrics (I_F)
    "I_F_xOnGoal", "I_F_xGoals", "I_F_xRebounds", "I_F_xFreeze", "I_F_xPlayStopped", "I_F_xPlayContinuedInZone",
    "I_F_xPlayContinuedOutsideZone", "I_F_flurryAdjustedxGoals", "I_F_scoreVenueAdjustedxGoals",
    "I_F_flurryScoreVenueAdjustedxGoals", "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_shotsOnGoal",
    "I_F_missedShots", "I_F_blockedShotAttempts", "I_F_shotAttempts", "I_F_points", "I_F_goals", "I_F_rebounds",
    "I_F_reboundGoals", "I_F_freeze", "I_F_playStopped", "I_F_playContinuedInZone", "I_F_playContinuedOutsideZone",
    "I_F_savedShotsOnGoal", "I_F_savedUnblockedShotAttempts", "I_F_penalityMinutes", "I_F_faceOffsWon", "I_F_hits",
    "I_F_takeaways", "I_F_giveaways", "I_F_lowDangerShots", "I_F_mediumDangerShots", "I_F_highDangerShots",
    "I_F_lowDangerxGoals", "I_F_mediumDangerxGoals", "I_F_highDangerxGoals", "I_F_lowDangerGoals", "I_F_mediumDangerGoals",
    "I_F_highDangerGoals", "I_F_scoreAdjustedShotsAttempts", "I_F_unblockedShotAttempts",
    "I_F_scoreAdjustedUnblockedShotAttempts", "I_F_dZoneGiveaways", "I_F_xGoalsFromxReboundsOfShots",
    "I_F_xGoalsFromActualReboundsOfShots", "I_F_reboundxGoals", "I_F_xGoals_with_earned_rebounds",
    "I_F_xGoals_with_earned_rebounds_scoreAdjusted", "I_F_xGoals_with_earned_rebounds_scoreFlurryAdjusted",
    "I_F_shifts", "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts", "I_F_neutralZoneShiftStarts", "I_F_flyShiftStarts",
    "I_F_oZoneShiftEnds", "I_F_dZoneShiftEnds", "I_F_neutralZoneShiftEnds", "I_F_flyShiftEnds",

    # On-Ice For Stats (when player is on ice)
    "OnIce_F_xOnGoal", "OnIce_F_xGoals", "OnIce_F_flurryAdjustedxGoals", "OnIce_F_scoreVenueAdjustedxGoals",
    "OnIce_F_flurryScoreVenueAdjustedxGoals", "OnIce_F_shotsOnGoal", "OnIce_F_missedShots", "OnIce_F_blockedShotAttempts",
    "OnIce_F_shotAttempts", "OnIce_F_goals", "OnIce_F_rebounds", "OnIce_F_reboundGoals", "OnIce_F_lowDangerShots",
    "OnIce_F_mediumDangerShots", "OnIce_F_highDangerShots", "OnIce_F_lowDangerxGoals", "OnIce_F_mediumDangerxGoals",
    "OnIce_F_highDangerxGoals", "OnIce_F_lowDangerGoals", "OnIce_F_mediumDangerGoals", "OnIce_F_highDangerGoals",
    "OnIce_F_scoreAdjustedShotsAttempts", "OnIce_F_unblockedShotAttempts", "OnIce_F_scoreAdjustedUnblockedShotAttempts",
    "OnIce_F_xGoalsFromxReboundsOfShots", "OnIce_F_xGoalsFromActualReboundsOfShots", "OnIce_F_reboundxGoals",
    "OnIce_F_xGoals_with_earned_rebounds", "OnIce_F_xGoals_with_earned_rebounds_scoreAdjusted",
    "OnIce_F_xGoals_with_earned_rebounds_scoreFlurryAdjusted",

    # On-Ice Against Stats (when player is on ice)
    "OnIce_A_xOnGoal", "OnIce_A_xGoals", "OnIce_A_flurryAdjustedxGoals", "OnIce_A_scoreVenueAdjustedxGoals",
    "OnIce_A_flurryScoreVenueAdjustedxGoals", "OnIce_A_shotsOnGoal", "OnIce_A_missedShots", "OnIce_A_blockedShotAttempts",
    "OnIce_A_shotAttempts", "OnIce_A_goals", "OnIce_A_rebounds", "OnIce_A_reboundGoals", "OnIce_A_lowDangerShots",
    "OnIce_A_mediumDangerShots", "OnIce_A_highDangerShots", "OnIce_A_lowDangerxGoals", "OnIce_A_mediumDangerxGoals",
    "OnIce_A_highDangerxGoals", "OnIce_A_lowDangerGoals", "OnIce_A_mediumDangerGoals", "OnIce_A_highDangerGoals",
    "OnIce_A_scoreAdjustedShotsAttempts", "OnIce_A_unblockedShotAttempts", "OnIce_A_scoreAdjustedUnblockedShotAttempts",
    "OnIce_A_xGoalsFromxReboundsOfShots", "OnIce_A_xGoalsFromActualReboundsOfShots", "OnIce_A_reboundxGoals",
    "OnIce_A_xGoals_with_earned_rebounds", "OnIce_A_xGoals_with_earned_rebounds_scoreAdjusted",
    "OnIce_A_xGoals_with_earned_rebounds_scoreFlurryAdjusted",

    # Off-Ice Stats (when player is not on ice)
    "OffIce_F_xGoals", "OffIce_A_xGoals", "OffIce_F_shotAttempts", "OffIce_A_shotAttempts"
]

nonnum_columns = ["playerId", "name", "position", "season"]
numeric_columns  = list(set(metrics_no_icetime) - set(nonnum_columns))