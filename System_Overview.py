





# Proposed ARC-Solver System Overview

for task in tasks

    list_of_IN_segs_by_strat_by_example  = []
    list_of_OUT_segs_by_strat_by_example = []

    for ex_num, example in enumerate(examples)

        list_of_IN_segs_by_strat_for_curr_example  = []
        list_of_OUT_segs_by_strat_for_curr_example = []

        # Get the IN and OUT json grids for this example
        IN = example['input']
        OUT = example['output']

        # Get the segmentation strategies that were successful in the past for grids of this type (if none, then rand)
        k_best_seg_strats_for_IN = best_seg_strats_for_grid(IN, k)
        k_best_seg_strats_for_OUT = best_seg_strats_for_grid(OUT, k)

        for segment_strategy in k_best_seg_strats_for_IN
            # Get list of segments of the IN grid using the segmentation strategy
            IN_segs_for_strat = get_segments_list(IN, segment_strategy) # segment_strategy = num_slots, …, etc.
            list_of_IN_segs_by_strat_for_curr_example.append(IN_segs_for_strat) # Append list of segs to list of lists
        
        list_of_IN_segs_by_strat_by_example.append(list_of_IN_segs_by_strat_for_curr_example)

        for segment_strategy in k_best_seg_strats_for_OUT
            # Get list of segments of the OUT grid using the segmentation strategy
           OUT_segs_for_strat = get_segments_list(OUT, segment_strategy) # segment_strategy = num_slots, …, etc.
           list_of_OUT_segs_by_strat_for_curr_example.append(OUT_segs_for_strat) # Append list of segs to list of lists



    # Now we have a set of IN and OUT segments for each example, across seg strategies
    # Let's determine which segment strategies give the most consistent results across examples
    # Consistent means: 
    # 1) Similar number of pixels in segments, across all examples. 
    #    Inconsistent: a less consistent strategy will have for a case of a 4pix object present in two examples, 
    #    one example with segs of 1pix and 3pix, and another example with and 2pix and 2pix for another example
    #    A consitent segmentation strategy will have 2pix and 2pix for both examples
    #    ...(assuming these two objects are equivalent across the examples...)
    #   MORE? ...

    # Segment objects/images should also maintain relative positions in overall grid...

    # The lists "list_of_<IN/OUT>_segs_by_strat_for_curr_example" are: [example, strat, [segs]]
    # For each strategy, see which has segments are most consistent across examples
    for strat in list_of_IN_segs_by_strat_by_example


        DICT instead of list???
        " index-based dictionary is the best choice"

        Answer: DOES applying prims to segs/objects/abstractions, and/or compressing them into one another
        (making new prim) actually solve task..??

        # Sort strats by consistancy: more to less similarity of their seg sets across examples

    # E.g.
    # [example1, strat1, [s111, s112, s113]]
    # [example1, strat2, [s121, s122, s123]]
    # [example1, strat3, [s131, s132, s133]]
    # [example2, strat1, [s211, s212, s213]]
    


    most_consistent_seg_strategies = ...

    # Conjecture that this segmentation strategy is the best for this TYPE of task:
    # Abstract the set of examples and the segmentation strategies. Associate with one another...   TODO: What means of associating?
    encode(example_set)
    for segmentation_strategy in most_consistent_seg_strategies:
        encode(segmentation_strategy)
        associate(example_set, segmentation_strategy)


    # Take the sets of segments from these most consistent strategies (maintaining I to O correspondence)
    seg_sets_from_best_strategies = ...

    for segment_set in seg_sets_from_best_strategies:
    # For each segment in the set find the most promising primitives
        for seg in segment_set:
            # Get the primitives that have been successful in the past for grids of this type (if none, then rand)
            # "Successful" primitives are those that contributed to achieving a task solution. Handled below. 
            # TODO: OR those which produce coherent results... n.n. just task success.
            k_best_prims_for_seg = best_prims_for_seg(seg)

            for prim in k_best_prims_for_seg:
                # Apply the primitive to the segment and get the resulting abstractions, however many
                abstractions = apply_prim(prim, seg)

            # We've now got a set of abstractions for each segment in the set


    For all abstractions, from I and O, see which are closest...   ///  BUt closest not necessarily the right track...

    ... arrange, combine, compose ........
    I.e. make conjectures as to the *solution* to the task. If refuted, mutate, recombine, etc. etc. or scrap altogether.

    Surprisal? See chat.

# As before with segmentation strategies and examples, now associate primitives in the conjecture which contributed to success 
# with the segments they acted on, AND WITH THE TASK ITSELF.....   again TODO: What means of associating?

