# Single engineering correction

Inference completed all12forwards with exact normalized-input equivalence checks and prediction freeze. The first analysis process wrote all CSVs, reference RLEs and COMPLETE.json successfully, then exited with a TypeError while printing its console JSON because a paired count was numpy.int64. The complete traceback is retained in analysis.log.

Correct only the final console serialization with numpy scalar conversion. No model rerun, deletion/replacement of completed metrics, change of cohort, target, score, transform, assignment or threshold. The saved results are read directly for reporting and independently checked against saved RLEs. A nonzero console exit must not be hidden or treated as absent scientific outputs. Actual verification, not the existence of COMPLETE.json alone, determines whether outputs are usable.
